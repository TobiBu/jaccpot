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
   that is rejected itself is a near leaf. Dense over groups x nodes, chunked in
   blocks of groups; the lists are compacted to the plan's fixed capacities.
4. **Far:** M2P per point -- M2L of the used nodes onto the point itself, L2P at
   zero offset (the lane's own real-basis operators); a local expansion about the
   group centre would add a group-radius truncation error to every contribution,
   coherently, which eq (16a)'s per-interaction control does not bound. **Near:** the softened pair kernel of the
   engine (:func:`~jaccpot.softening.masked_pair_factors`) over the near leaves'
   particles; a point on top of a source gets the kernel's finite value.

Everything is traced and static-shaped: the :class:`TargetPlan` (group size,
far/near capacities) is the only static input, so moving points recompile
nothing. An overflow of either capacity is reported (``ok``, ``needs``), never
silent; :func:`plan_target_points` sizes the capacities eagerly.

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
from typing import Any, NamedTuple, Optional

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

    Attributes
    ----------
    num_points : int
        ``M``, the number of target points.
    group_size : int
        Points per target group.
    far_capacity : int
        Far (M2L) source nodes per group.
    near_capacity : int
        Near (P2P) source leaves per group.
    group_block : int
        Groups walked together (memory: ``group_block x nodes`` booleans).
    """

    num_points: int
    group_size: int = 32
    far_capacity: int = 256
    near_capacity: int = 64
    group_block: int = 64

    def __post_init__(self) -> None:
        for name in ("num_points", "group_size", "far_capacity", "near_capacity"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"TargetPlan.{name} must be >= 1")
        if int(self.group_block) < 1:
            raise ValueError("TargetPlan.group_block must be >= 1")

    @property
    def num_groups(self) -> int:
        """Groups after padding ``M`` up to a multiple of ``group_size``."""
        return -(-int(self.num_points) // int(self.group_size))

    def grown(self, needs: Any, headroom: float = 1.5) -> "TargetPlan":
        """A plan whose capacities cover ``needs`` with ``headroom``.

        Parameters
        ----------
        needs : Any
            ``(far, near)`` the largest counts a group needed.
        headroom : float
            Multiplier on the needs.

        Returns
        -------
        TargetPlan
            The same plan with capacities ``max(old, ceil8(need * headroom))``.
        """
        far, near = (int(v) for v in np.asarray(needs).reshape(-1)[:2])
        return dataclasses.replace(
            self,
            far_capacity=max(int(self.far_capacity), _ceil8(far * headroom)),
            near_capacity=max(int(self.near_capacity), _ceil8(near * headroom)),
        )


class TargetField(NamedTuple):
    """What a target evaluation returns (in the points' order and dtype).

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
        ``int32 [2]``: the largest far / near count any group needed.
    ok : Array
        Boolean scalar: both counts fit the plan (the result is then complete).
    """

    acc: Array
    potential: Optional[Array]
    force_scale: Optional[Array]
    needs: Array
    ok: Array


def _ceil8(value: float) -> int:
    """``value`` rounded up to a multiple of 8, at least 8.

    Parameters
    ----------
    value : float
        A count.

    Returns
    -------
    int
        The rounded count.
    """
    return max(8, int(8 * math.ceil(float(value) / 8.0)))


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


def _evaluate_group(
    src: _SourceTables,
    points: Array,
    valid: Array,
    threshold: Optional[Array],
    *,
    plan: TargetPlan,
    mac: str,
    theta: float,
    theta_max: float,
    floor: float,
    order: int,
    leaf_cap: int,
    G: float,
    soft_params: Array,
    kernel: str,
    eps2_plummer: float,
    want_potential: bool,
    want_scale: bool,
) -> tuple[Array, Array, Array, Array]:
    """One group's ``(acc [G,3], pot [G], f_b [G], counts [2])``.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    points : Array
        ``[G, 3]`` the group's points (padding repeats a live point).
    valid : Array
        ``[G]`` live points.
    threshold : Optional[Array]
        The group's eq (16a) threshold.
    plan : TargetPlan
        Static capacities.
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
    tuple[Array, Array, Array, Array]
        Acceleration, potential, force scale, and the far / near counts.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.operators.real_p2m_l2p import evaluate_local_real_with_grad
    from jaccpot.softening import masked_pair_factors

    dtype = src.radius.dtype
    nodes = int(src.radius.shape[0])
    lo = jnp.min(jnp.where(valid[:, None], points, jnp.inf), axis=0)
    hi = jnp.max(jnp.where(valid[:, None], points, -jnp.inf), axis=0)
    centre = 0.5 * (lo + hi)
    rad = jnp.sqrt(
        jnp.max(jnp.where(valid, jnp.sum((points - centre) ** 2, axis=1), 0.0))
    )
    far, near = _walk_masks(
        src,
        centre,
        rad,
        threshold,
        mac=mac,
        theta=theta,
        theta_max=theta_max,
        floor=floor,
        order=order,
    )
    counts = jnp.stack([jnp.sum(far), jnp.sum(near)]).astype(jnp.int32)
    far_ids = jnp.nonzero(far, size=int(plan.far_capacity), fill_value=nodes)[0]
    near_ids = jnp.nonzero(near, size=int(plan.near_capacity), fill_value=nodes)[0]

    # ---- far: M2P per point (M2L onto the point, L2P at zero offset) --------
    # A local expansion about the GROUP's centre would add its truncation error
    # (the group radius over the distance, to the power p + 1) to every source
    # node's contribution, coherently: measured 5e-4 rel at eps 1e-4 on sparse
    # groups, whatever the source nodes. Per point, only the multipoles' own
    # truncation remains, which the criterion above bounds.
    mult = src.multipoles[far_ids]
    centres = src.centers[far_ids]
    live = far_ids < nodes

    def m2p(x: Array) -> tuple[Array, Array]:
        deltas = x[None, :] - centres
        deltas = jnp.where(live[:, None], deltas, jnp.ones_like(deltas))
        locals_ = m2l_rot_scale_real_batch(mult, deltas, order=int(order))
        local = jnp.sum(jnp.where(live[:, None], locals_, 0.0), axis=0)
        grad, phi = evaluate_local_real_with_grad(
            local, jnp.zeros((3,), dtype), order=int(order)
        )
        return grad, phi

    grad, phi = jax.vmap(m2p)(points)
    gconst = jnp.asarray(G, dtype)
    acc = -gconst * grad
    pot = -gconst * phi
    scale = jnp.zeros(points.shape[:1], dtype)
    if want_scale:
        dc = src.centers[far_ids][None, :, :] - points[:, None, :]
        r2c = jnp.sum(dc * dc, axis=-1) + jnp.asarray(eps2_plummer, dtype)
        scale = gconst * jnp.sum(
            jnp.where(live[None, :], src.mass[far_ids][None, :] / r2c, 0.0), axis=1
        )

    # ---- near: softened pairs with the near leaves' particles ---------------
    start = src.ranges[near_ids, 0]
    count = src.ranges[near_ids, 1] - start + 1
    slot = jnp.arange(int(leaf_cap), dtype=jnp.int32)
    sidx = (start[:, None] + slot[None, :]).reshape(-1)
    slive = (slot[None, :] < count[:, None]).reshape(-1)
    sidx = jnp.where(slive, sidx, 0)
    spos = src.positions[sidx]
    smas = jnp.where(slive, src.masses[sidx], 0.0)
    dx = points[:, None, :] - spos[None, :, :]
    r2 = jnp.sum(dx * dx, axis=-1)
    active = slive[None, :] & valid[:, None]
    g, psi, _ = masked_pair_factors(
        r2, active, soft_params, kernel, potential=want_potential
    )
    acc = acc - gconst * jnp.sum((smas[None, :] * g)[..., None] * dx, axis=1)
    if want_potential:
        pot = pot - gconst * jnp.sum(smas[None, :] * psi, axis=1)
    if want_scale:
        r2s = jnp.where(active, r2 + jnp.asarray(eps2_plummer, dtype), 1.0)
        scale = scale + gconst * jnp.sum(
            jnp.where(active, smas[None, :] / r2s, 0.0), axis=1
        )
    return acc, pot, scale, counts


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
    pts = pts_in.astype(dtype)
    m = int(plan.num_points)
    gsize = int(plan.group_size)
    ngroups = plan.num_groups
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
    from jaccpot.softening import resolve_softening_kernel, softening_params

    kernel = resolve_softening_kernel(getattr(engine, "softening_kernel", None))
    eps = float(engine.softening)
    statics = dict(
        plan=plan,
        mac=mac_name,
        theta=float(provider.theta_value),
        theta_max=float(getattr(engine, "mac_theta_max", 1.0)),
        floor=float(engine._walk_separation_floor()),
        order=order,
        leaf_cap=int(topology.prepared.max_leaf_size),
        G=float(engine.G),
        soft_params=softening_params(kernel, eps, dtype),
        kernel=kernel,
        eps2_plummer=eps * eps,
        want_potential=bool(potential),
        want_scale=bool(return_force_scale),
    )

    def one(args: tuple) -> tuple:
        gp, gv, gt = args
        return _evaluate_group(src, gp, gv, gt, **statics)

    acc, pot, scale, counts = jax.lax.map(
        one, (groups, gvalid, thr), batch_size=min(int(plan.group_block), ngroups)
    )
    needs = jnp.max(counts, axis=0)
    ok = (needs[0] <= int(plan.far_capacity)) & (needs[1] <= int(plan.near_capacity))
    inv = jnp.argsort(perm)
    out_dtype = pts_in.dtype

    def back(v: Array, tail: tuple) -> Array:
        return v.reshape((ngroups * gsize,) + tail)[:m][inv].astype(out_dtype)

    return TargetField(
        acc=back(acc, (3,)),
        potential=back(pot, ()) if potential else None,
        force_scale=back(scale, ()) if return_force_scale else None,
        needs=needs,
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
        Groups walked together.
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
        far_capacity=8,
        near_capacity=8,
        group_block=int(group_block),
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
