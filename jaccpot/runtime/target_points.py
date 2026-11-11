"""The field at points that are not sources: ``target_points`` (issue #302).

The fused lane evaluates its force at its own sources. A caller that needs the
field somewhere else -- the galaxy's tidal field at the stars of a globular
cluster that is integrated by direct summation (hybrid Hermite, multi-node plan
U6), an observation operator's tracers -- had to append the points as zero-mass
particles, which breaks the mass-dependent criterion, takes tree capacity and
moves the topology. This module evaluates the field of a :class:`FusedTopology`'s
sources at ``M`` massless points instead, with the topology untouched:

1. **Upward pass.** The topology keeps its local expansions, not its multipoles,
   so the sources' P2M/M2M is run again on the topology's tree
   (``prepare_upward_sweep``: the rebuild's own sweep, about the COMs) with the
   walk's exact COM radii (:func:`~jaccpot.runtime._mac_geometry.com_mac_geometry`).
2. **Target groups.** The points are sorted along a Morton curve of their own
   bounding box and cut into groups of ``group_size``; a group is a sphere (its
   box centre, the largest member distance). The groups are not a tree and do not
   enter the source tree: they exist only for the walk below.
3. **One-sided walk.** Every group tests every source node with the source walk's
   criterion: the geometric opening angle ``(r_n + r_g)^2 <= theta^2 d^2`` with the
   separation floor (a compact kernel's support), or eq (16a) of Dehnen (2014)
   for ``mac_type='dehnen_error'`` -- the SOURCE node's multipole error against
   the GROUP's threshold ``adaptive_eps * min_b f_b`` over the group's own points
   (their f_b, never the sources'), with eq (15) for M2P (``rho_B = 0``) at the
   distance of the group's closest possible member. A node is used if it is accepted and every
   ancestor was rejected (pointer doubling over the parent array, so the
   criterion need not be monotone); a leaf whose ancestors were all rejected and
   that is rejected itself is a near leaf. The walk is dense over groups x nodes
   (O(groups x nodes) work), run in blocks of groups sized so a block's masks fit
   ``_WALK_BUDGET_BYTES``; each block's accepted pairs are scattered into FLAT
   (group, node) far and (group, leaf) near lists at a running offset, whose
   lengths are the plan's TOTAL capacities.
4. **Far:** M2P per point -- M2L of the used nodes onto the point itself, L2P at
   zero offset (the lane's own real-basis operators); a local expansion about the
   group centre would add a group-radius truncation error to every contribution,
   coherently, which eq (16a)'s per-interaction control does not bound.
   **Near:** the softened pair kernel of the engine
   (:func:`~jaccpot.softening.masked_pair_factors`) over the near leaves'
   particles; a point on top of a source gets the kernel's finite value. Both
   run ``plan.pair_chunk`` pairs at a time and scatter-add into the points.

Everything is traced and static-shaped: the :class:`TargetPlan` (group size,
total far/near capacities, chunk) is the only static input, so moving points
recompile nothing. An overflow of either capacity is reported (``ok``,
``needs``), never silent; :func:`plan_target_points` sizes the capacities
eagerly.

**Memory** is bounded by the flat lists (8 bytes per capacity entry), one walk
block (``_WALK_BUDGET_BYTES``) and one chunk of evaluation temporaries
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
    "bootstrap_target_force_scale",
    "evaluate_target_points",
    "plan_target_points",
    "target_mac_type",
]

#: the criteria the target walk implements
_GEOMETRIC_MACS = ("dehnen", "bh")
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
    """

    num_points: int
    group_size: int = 32
    far_capacity: int = 4096
    near_capacity: int = 1024
    group_block: int = 64
    pair_chunk: int = 128

    def __post_init__(self) -> None:
        for name in (
            "num_points",
            "group_size",
            "far_capacity",
            "near_capacity",
            "group_block",
            "pair_chunk",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"TargetPlan.{name} must be >= 1")

    @property
    def num_groups(self) -> int:
        """Groups after padding ``M`` up to a multiple of ``group_size``."""
        return -(-int(self.num_points) // int(self.group_size))

    def grown(self, needs: Any, headroom: float = 1.5) -> "TargetPlan":
        """A plan whose capacities cover ``needs`` with ``headroom``.

        Parameters
        ----------
        needs : Any
            ``(far, near)`` total pair counts an evaluation needed.
        headroom : float
            Multiplier on the needs.

        Returns
        -------
        TargetPlan
            The same plan with capacities ``max(old, need * headroom)``, rounded
            up to a multiple of ``pair_chunk``.
        """
        far, near = (int(v) for v in np.asarray(needs).reshape(-1)[:2])
        chunk = int(self.pair_chunk)
        return dataclasses.replace(
            self,
            far_capacity=max(int(self.far_capacity), _ceil_to(far * headroom, chunk)),
            near_capacity=max(
                int(self.near_capacity), _ceil_to(near * headroom, chunk)
            ),
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
        Boolean scalar: both counts fit the plan (the result is then complete).
    """

    acc: Array
    potential: Optional[Array]
    force_scale: Optional[Array]
    needs: Array
    ok: Array


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


def _morton_order(points: Array) -> Array:
    """Argsort of the points along a 10-bit-per-axis Morton curve of their box.

    Parameters
    ----------
    points : Array
        ``[M, 3]``.

    Returns
    -------
    Array
        ``[M]`` the stable argsort of the Morton codes.
    """
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    span = jnp.maximum(jnp.max(hi - lo), jnp.asarray(_TINY, points.dtype))
    q = jnp.clip(((points - lo) / span) * 1023.0, 0.0, 1023.0).astype(jnp.uint32)

    def spread(v: Array) -> Array:
        v = (v | (v << 16)) & jnp.uint32(0x030000FF)
        v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
        v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
        v = (v | (v << 2)) & jnp.uint32(0x09249249)
        return v

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
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
        ``[nodes + 1, (p+1)^2]`` packed real multipoles (sentinel row zero).
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

    from ._mac_geometry import com_mac_geometry
    from ._walk_criterion import dehnen_walk_table

    prepared = topology.prepared
    tree = prepared.tree
    pos = jnp.asarray(tree.positions_sorted)
    mas = jnp.asarray(tree.masses_sorted, pos.dtype)
    leaf_cap = int(prepared.max_leaf_size)
    num_levels = engine._resolve_upward_num_levels(tree)
    upward = engine.prepare_upward_sweep(
        tree,
        pos,
        mas,
        max_order=int(order),
        center_mode="com",
        max_leaf_size=leaf_cap,
        defer_geometry=True,
    )
    centers = jnp.asarray(upward.multipoles.centers, pos.dtype)
    packed = jnp.asarray(upward.multipoles.packed)
    radius = com_mac_geometry(
        tree, pos, centers, leaf_cap=leaf_cap, num_levels=num_levels
    ).radius
    radius = jnp.asarray(radius, pos.dtype) * jnp.asarray(
        float(getattr(engine, "dehnen_radius_scale", 1.0) or 1.0), pos.dtype
    )
    mass = jnp.asarray(upward.mass_moments.mass, pos.dtype)
    ranges = jnp.asarray(tree.node_ranges, jnp.int32)
    nodes = int(ranges.shape[0])
    internal = int(jnp.asarray(tree.left_child).shape[0])
    counts = ranges[:, 1] - ranges[:, 0] + 1
    parent = jnp.asarray(tree.parent, jnp.int32)
    parent = jnp.where((parent < 0) | (parent >= nodes), nodes, parent)
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
        multipoles=jnp.concatenate(
            [packed, jnp.zeros((1, packed.shape[1]), packed.dtype)]
        ),
        mass=jnp.concatenate([mass, jnp.zeros((1,), pos.dtype)]),
        nonempty=(counts > 0) & (mass != 0.0),
        is_leaf=jnp.arange(nodes) >= internal,
        parent=jnp.concatenate([parent, jnp.asarray([nodes], jnp.int32)]),
        ranges=jnp.concatenate([ranges, jnp.asarray([[0, -1]], jnp.int32)]),
        table=table,
        positions=pos,
        masses=mas,
    )


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
) -> tuple[Array, Array]:
    """``(far, near)`` node masks of ONE group against every source node.

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

    Returns
    -------
    tuple[Array, Array]
        ``[nodes]`` far (M2P) and near (P2P leaf) masks.
    """
    from yggdrax.interactions import WALK_TABLE_POWER0, dehnen_pair_accept

    nodes = int(src.radius.shape[0])
    dtype = src.radius.dtype
    diff = src.centers[:nodes] - centre[None, :]
    dist_sq = jnp.sum(diff * diff, axis=1)
    valid = src.nonempty & (dist_sq > 0.0)
    if float(floor) > 0.0:
        reach = src.radius + rad + jnp.asarray(floor, dtype)
        valid = valid & (reach * reach <= dist_sq)
    if mac == "dehnen_error":
        # the far field is M2P (no local expansion about the group), so eq (15)
        # is taken with rho_B = 0 at the distance of the group's closest possible
        # member, d - r_g: an upper bound for every point of the group
        cols = WALK_TABLE_POWER0 + int(order)
        row_a = [src.table[:, k] for k in range(cols)]
        zero = jnp.zeros((nodes,), dtype)
        row_b = [zero, jnp.broadcast_to(threshold, (nodes,))] + [
            zero for _ in range(cols - 2)
        ]
        closest = jnp.maximum(jnp.sqrt(dist_sq) - rad, 0.0)
        accept = valid & dehnen_pair_accept(
            row_a=row_a,
            row_b=row_b,
            radius_a=src.radius,
            radius_b=zero,
            dist_sq=closest * closest,
            order=int(order),
            theta_max=jnp.asarray(theta_max, dtype),
        )
    else:
        extent = src.radius + rad
        accept = valid & (
            extent * extent <= jnp.asarray(theta * theta, dtype) * dist_sq
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
        src.multipoles[node][:, None, :], (chunk, gsize, src.multipoles.shape[1])
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
) -> TargetField:
    """The field of ``topology``'s sources at ``points``; traced or eager.

    Three passes, every one memory-bounded: the walk, block by block, compacted
    into flat (group, node) far and (group, leaf) near lists of the plan's
    total capacities; then the far pairs (M2P) and the near pairs (softened
    P2P), ``plan.pair_chunk`` at a time, scatter-added into the groups' points.

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

    Returns
    -------
    TargetField
        In the points' order and dtype.

    Raises
    ------
    ValueError
        If ``points`` is not ``[plan.num_points, 3]``, or ``dehnen_error`` has no
        ``force_scale``.
    """
    from jaccpot.softening import resolve_softening_kernel, softening_params

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
    order = int(provider.max_order)
    src = _source_tables(engine, topology, order=order, mac=mac_name)
    dtype = src.radius.dtype
    nodes = int(src.radius.shape[0])
    block = _walk_block(plan, nodes)
    pts = pts_in.astype(dtype)
    m = int(plan.num_points)
    gsize = int(plan.group_size)
    ngroups = -(-plan.num_groups // block) * block
    pad = ngroups * gsize - m
    perm = _morton_order(pts)
    sorted_pts = pts[perm]
    valid = jnp.arange(ngroups * gsize) < m
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
    fg, fn, ng, nn, totals = _compact_pairs(
        src,
        centres,
        radii,
        thr,
        live_groups,
        block=block,
        far_cap=far_cap,
        near_cap=near_cap,
        walk=dict(
            mac=mac_name,
            theta=float(provider.theta_value),
            theta_max=float(getattr(engine, "mac_theta_max", 1.0)),
            floor=float(engine._walk_separation_floor()),
            order=order,
        ),
    )
    kernel = resolve_softening_kernel(getattr(engine, "softening_kernel", None))
    eps = float(engine.softening)
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

    carry = (zeros3, zeros1, zeros1)
    carry, _ = jax.lax.scan(
        far_body, carry, (fg.reshape(-1, chunk), fn.reshape(-1, chunk))
    )
    (acc, pot, scale), _ = jax.lax.scan(
        near_body, carry, (ng.reshape(-1, chunk), nn.reshape(-1, chunk))
    )
    ok = (totals[0] <= far_cap) & (totals[1] <= near_cap)
    inv = jnp.argsort(perm)
    out_dtype = pts_in.dtype

    def back(v: Array, tail: tuple) -> Array:
        return v.reshape((ngroups * gsize,) + tail)[:m][inv].astype(out_dtype)

    return TargetField(
        acc=back(acc, (3,)),
        potential=back(pot, ()) if potential else None,
        force_scale=back(scale, ()) if return_force_scale else None,
        needs=totals,
        ok=ok,
    )


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
        ``jit(run)(topology, points, force_scale) -> TargetField``.
    """

    def run(topology: Any, points: Array, force_scale: Optional[Array]) -> TargetField:
        return evaluate_target_points(
            provider, topology, points, plan=plan, force_scale=force_scale, **static
        )

    return jax.jit(run)


def bootstrap_target_force_scale(
    provider: Any, topology: Any, points: Array, *, plan: TargetPlan
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

    Returns
    -------
    tuple[Array, TargetField]
        ``(f_b, field)`` of a pass with the geometric criterion at the engine's
        theta.
    """
    field = _jitted(provider, plan, return_force_scale=True, mac="dehnen")(
        topology, points, None
    )
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
    max_rounds: int = 4,
) -> TargetPlan:
    """Size a :class:`TargetPlan` eagerly from the points' current needs.

    Starts from small capacities and grows them from the evaluation's ``needs``
    until it fits, then adds ``headroom`` (moving points stay inside it; a step
    that does not reports ``ok=False``).

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        A topology at the sources' current positions.
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
        Upper bound on the groups walked together.
    pair_chunk : int
        Pairs evaluated together.
    max_rounds : int
        Re-plans before giving up.

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
    )
    if target_mac_type(provider.engine) == "dehnen_error" and force_scale is None:
        for _ in range(int(max_rounds)):
            fb, field = bootstrap_target_force_scale(
                provider, topology, points, plan=plan
            )
            if bool(field.ok):
                break
            plan = plan.grown(field.needs, 1.0)
        force_scale = fb
    for _ in range(int(max_rounds)):
        field = _jitted(provider, plan)(topology, points, force_scale)
        if bool(field.ok):
            return plan.grown(field.needs, float(headroom))
        plan = plan.grown(field.needs, 1.0)
    raise RuntimeError(
        f"plan_target_points: no plan fitted in {max_rounds} rounds (last needs "
        f"{np.asarray(field.needs).tolist()})"
    )
