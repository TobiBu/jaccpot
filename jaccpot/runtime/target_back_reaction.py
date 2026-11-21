"""A compact source set's field at very many points: the hybrid back-reaction.

The hybrid Hermite scheme (multi-node plan U6) needs the field of a star
cluster (``M ~ 1e4-1e6`` stars, a small tree of its own) at every galaxy
particle (``N ~ 1e8``). :func:`~jaccpot.runtime.target_points.evaluate_target_points`
over all ``N`` points would sort, group and walk them all (session 1 measured
~10-11 GB of per-point transients at 1e8). Almost every galaxy particle is far
from the cluster, where the cluster's ROOT multipole alone is accurate, so:

* **Outer points** -- ``|x - c| >= r_in`` with ``c`` the cluster's COM (the root's
  expansion centre) and ``r_in = max(r_min, R / theta_mp, R + floor)``, ``R`` the
  root's exact COM radius (every star lies inside it) and ``floor`` the target
  walk's separation floor at the points' softening: the root multipole,
  truncated at ``mp_order``, evaluated per point (M2P), slice by slice
  (``slice_size`` points at a time, a ``fori_loop``: no sort, no groups, no
  per-point lists). ``R / theta_mp`` is the target walk's own acceptance of the
  root for a single point under the geometric angle, so with
  ``theta_mp = theta`` and ``mp_order = p`` (the defaults) these points get
  exactly what the walk would give them; the floor keeps a compact kernel's
  support outside (the field there is Newtonian).
* **Inner points** -- the rest, compacted into a static capacity
  (``inner_capacity``) and evaluated by ``target_points`` with a ``point_mask``
  (dead slots cost nothing). An inner count above the capacity is reported
  (``ok=False``; the stats carry the count), never silent.

**K source groups** (:class:`SourceGroups`, ``plan.num_groups = K``): when the
sources are K compact clusters spread over the points (globular clusters all
over a galaxy) the root multipole is useless -- its ``r_in`` covers everything.
Each group ``k`` then gets its own multipole about its own COM ``c_k`` (a
direct P2M over its members, truncated at ``mp_order``), its exact radius
``R_k`` (the farthest member) and its own ``r_in_k``. A point outside EVERY
group's ``r_in_k`` takes the sum of the K multipoles (M2P, ``group_chunk``
groups at a time); a point inside ANY is an inner point and takes the whole
source tree through the target walk (every group's field, at the walk's
accuracy), so no group is counted twice. A group of zero mass (a dissolved
cluster) contributes nothing and has no inner region.

A ``point_mask`` (``False``: an inactive point, e.g. a free row of a capacity
reserve) gives a point zero field and keeps it out of the inner walk, whose
groups are then those of the active points alone: inactive points change no
active point's field (bitwise).

Everything is traced and static-shaped; :func:`plan_back_reaction` sizes the
capacities eagerly. The result is a
:class:`~jaccpot.runtime.target_points.TargetField` (``needs`` the inner walk's
``int32 [2]``), so a caller of ``provider.target_field`` swaps the call:

.. code-block:: python

    # prepare
    plan_g = plan_back_reaction(clu, ctopo, xg, softening=eps_g)
    # each step
    back = evaluate_back_reaction(clu, ctopo, xg, plan=plan_g, softening=eps_g)
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .target_points import (
    TargetField,
    TargetPlan,
    _ceil_to,
    _count_dtype,
    _evaluate,
    _source_tables,
    plan_target_points,
    target_mac_type,
)

__all__ = [
    "BackReactionPlan",
    "SourceGroups",
    "evaluate_back_reaction",
    "plan_back_reaction",
]


@dataclasses.dataclass(frozen=True)
class BackReactionPlan:
    """Static shapes and the split of a back-reaction evaluation.

    Attributes
    ----------
    num_points : int
        ``N``, all the points.
    inner : TargetPlan
        The inner points' target plan; its ``num_points`` is the inner
        capacity.
    theta_mp : float
        The outer radius' opening angle: ``r_in >= R / theta_mp``.
    mp_order : int
        The root multipole's truncation order for the outer points.
    r_min : float
        A floor on ``r_in``.
    slice_size : int
        Outer points evaluated together.
    num_groups : int
        ``0``: the sources are one compact set (the root multipole, the
        session-2 split). ``K >= 1``: K :class:`SourceGroups` (each group's own
        multipole and inner region).
    group_chunk : int
        Groups whose multipoles one outer step evaluates together
        (``slice_size x group_chunk`` M2P pairs per step; grouped plans only).
    """

    num_points: int
    inner: TargetPlan
    theta_mp: float
    mp_order: int
    r_min: float = 0.0
    slice_size: int = 32768
    num_groups: int = 0
    group_chunk: int = 8

    def __post_init__(self) -> None:
        if int(self.num_points) < 1 or int(self.slice_size) < 1:
            raise ValueError("BackReactionPlan: num_points and slice_size >= 1")
        if int(self.mp_order) < 0:
            raise ValueError("BackReactionPlan.mp_order must be >= 0")
        if not 0.0 < float(self.theta_mp) < 1.0:
            raise ValueError("BackReactionPlan.theta_mp must be in (0, 1)")
        if float(self.r_min) < 0.0:
            raise ValueError("BackReactionPlan.r_min must be >= 0")
        if int(self.num_groups) < 0 or int(self.group_chunk) < 1:
            raise ValueError(
                "BackReactionPlan: num_groups must be >= 0 and group_chunk >= 1"
            )

    @property
    def inner_capacity(self) -> int:
        """Inner points the plan holds."""
        return int(self.inner.num_points)


class SourceGroups(NamedTuple):
    """K source groups (clusters) for a grouped back-reaction.

    The same particles the source topology was built from, laid out by group:
    ``positions[k, j]`` is member ``j`` of group ``k``; a zero mass is an empty
    slot (it belongs to no group). Traced (an argument of the evaluation).

    Attributes
    ----------
    positions : Array
        ``[K, M, 3]``.
    masses : Array
        ``[K, M]``.
    """

    positions: Array
    masses: Array


class _GroupTables(NamedTuple):
    """Per group: centre ``[K, 3]``, inner radius ``[K]``, multipole ``[K, (q+1)^2]``, mass ``[K]``."""

    centres: Array
    r_in: Array
    multipoles: Array
    mass: Array


def _group_tables(
    groups: SourceGroups,
    *,
    order: int,
    dtype: Any,
    theta_mp: float,
    r_min: float,
    floor: float,
) -> _GroupTables:
    """Each group's COM, exact radius, inner radius and multipole (direct P2M).

    The sums run in the wider of the groups' dtype and ``dtype`` and are cast to
    ``dtype``. A group without mass gets ``r_in = 0`` (no inner region) and a zero
    multipole.

    Parameters
    ----------
    groups : SourceGroups
        ``[K, M]`` members.
    order : int
        The multipoles' truncation order.
    dtype : Any
        The evaluation dtype.
    theta_mp : float
        ``r_in >= R / theta_mp``.
    r_min : float
        A floor on ``r_in``.
    floor : float
        ``r_in >= R + floor``.

    Returns
    -------
    _GroupTables
        The tables, in ``dtype``.
    """
    from jaccpot.operators.real_p2m_l2p import p2m_real_direct

    x = jnp.asarray(groups.positions)
    m = jnp.asarray(groups.masses)
    wide = jnp.result_type(x.dtype, m.dtype, dtype)
    x = x.astype(wide)
    m = m.astype(wide)
    filled = m > 0
    mass = jnp.sum(jnp.where(filled, m, 0.0), axis=1)
    live = mass > 0
    com = jnp.sum(jnp.where(filled[..., None], m[..., None] * x, 0.0), axis=1)
    com = jnp.where(live[:, None], com / jnp.where(live, mass, 1.0)[:, None], 0.0)
    d = jnp.where(filled[..., None], x - com[:, None, :], 0.0)
    big_r = jnp.sqrt(jnp.max(jnp.sum(d * d, axis=-1), axis=1))
    r_in = jnp.maximum(
        jnp.maximum(big_r / jnp.asarray(theta_mp, wide), big_r + floor),
        jnp.asarray(r_min, wide),
    )
    r_in = jnp.where(live, r_in, 0.0)

    def p2m(args: tuple) -> Array:
        dk, mk = args
        mk = jnp.where(mk > 0, mk, 0.0)
        return jnp.sum(
            jax.vmap(lambda dd, mm: p2m_real_direct(dd, mm, order=int(order)))(dk, mk),
            axis=0,
        )

    mult = jax.lax.map(p2m, (d, m))
    return _GroupTables(
        centres=com.astype(dtype),
        r_in=r_in.astype(dtype),
        multipoles=mult.astype(dtype),
        mass=mass.astype(dtype),
    )


def _floor(provider: Any, softening: Optional[float]) -> float:
    """The target walk's separation floor at the points' softening.

    Parameters
    ----------
    provider : Any
        The provider.
    softening : Optional[float]
        The points' softening (``None``: the engine's).

    Returns
    -------
    float
        The floor.
    """
    from jaccpot.softening import resolve_softening_kernel, support_factor

    engine = provider.engine
    if softening is None:
        return float(engine._walk_separation_floor())
    kernel = resolve_softening_kernel(getattr(engine, "softening_kernel", None))
    factor = getattr(engine, "softening_floor", None)
    return float(support_factor(kernel) if factor is None else factor) * float(
        softening
    )


def _outer_field(
    points: Array,
    centre: Array,
    r_in: Array,
    multipole: Array,
    mass: Array,
    *,
    order: int,
    G: float,
    eps2_plummer: float,
    slice_size: int,
    want_potential: bool,
    want_scale: bool,
    point_mask: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """The root multipole's field at the points outside ``r_in`` (zero inside).

    Parameters
    ----------
    points : Array
        ``[N, 3]``.
    centre : Array
        ``[3]`` the expansion centre.
    r_in : Array
        The inner radius.
    multipole : Array
        ``[(order+1)^2]`` the truncated root multipole.
    mass : Array
        The root's mass (eq 16b's monopole).
    order : int
        The truncation order.
    G : float
        Gravitational constant.
    eps2_plummer : float
        The Plummer-equivalent softening squared (eq 16b).
    slice_size : int
        Points per ``fori_loop`` step.
    want_potential : bool
        Also the potential.
    want_scale : bool
        Also eq (16b)'s force scale.
    point_mask : Optional[Array]
        ``[N]`` active points (``None``: all); inactive ones get zeros.

    Returns
    -------
    tuple[Array, Array, Array]
        ``[N, 3]`` acceleration, ``[N]`` potential, ``[N]`` force scale.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.operators.real_p2m_l2p import evaluate_local_real_with_grad

    n = int(points.shape[0])
    dtype = points.dtype
    size = min(int(slice_size), n)
    steps = -(-n // size)
    gconst = jnp.asarray(G, dtype)
    zero3 = jnp.zeros((3,), dtype)
    mult = jnp.asarray(multipole, dtype)

    def piece(x: Array, act: Optional[Array]) -> tuple[Array, Array, Array]:
        d = x - centre[None, :]
        r2 = jnp.sum(d * d, axis=-1)
        outer = r2 >= r_in * r_in
        if act is not None:
            outer = outer & act
        safe = jnp.where(outer[:, None], d, jnp.ones_like(d))
        loc = m2l_rot_scale_real_batch(
            jnp.broadcast_to(mult, (size, mult.shape[0])), safe, order=int(order)
        )
        grad, phi = jax.vmap(
            lambda c: evaluate_local_real_with_grad(c, zero3, order=int(order))
        )(loc)
        acc = jnp.where(outer[:, None], -gconst * grad, 0.0)
        pot = jnp.where(outer, -gconst * phi, 0.0) if want_potential else r2 * 0.0
        scale = (
            jnp.where(
                outer, gconst * mass / (r2 + jnp.asarray(eps2_plummer, dtype)), 0.0
            )
            if want_scale
            else r2 * 0.0
        )
        return acc.astype(dtype), pot.astype(dtype), scale.astype(dtype)

    def body(i: Array, carry: tuple) -> tuple:
        acc, pot, scale = carry
        start = jnp.minimum(i * size, n - size)
        x = jax.lax.dynamic_slice_in_dim(points, start, size, axis=0)
        act = (
            None
            if point_mask is None
            else jax.lax.dynamic_slice_in_dim(point_mask, start, size, axis=0)
        )
        da, dp, ds = piece(x, act)
        acc = jax.lax.dynamic_update_slice_in_dim(acc, da, start, axis=0)
        if want_potential:
            pot = jax.lax.dynamic_update_slice_in_dim(pot, dp, start, axis=0)
        if want_scale:
            scale = jax.lax.dynamic_update_slice_in_dim(scale, ds, start, axis=0)
        return acc, pot, scale

    init = (
        jnp.zeros((n, 3), dtype),
        jnp.zeros((n if want_potential else 1,), dtype),
        jnp.zeros((n if want_scale else 1,), dtype),
    )
    return jax.lax.fori_loop(0, steps, body, init)


def _outer_field_grouped(
    points: Array,
    tables: _GroupTables,
    *,
    order: int,
    G: float,
    eps2_plummer: float,
    slice_size: int,
    group_chunk: int,
    want_potential: bool,
    want_scale: bool,
    point_mask: Optional[Array],
) -> tuple[Array, Array, Array]:
    """The K groups' multipoles at the points outside EVERY group's ``r_in`` (zero elsewhere).

    ``slice_size`` points per ``fori_loop`` step; inside, a ``scan`` over chunks
    of ``group_chunk`` groups, each one batch of ``slice_size x group_chunk``
    M2P pairs (M2L onto the point, L2P at zero offset: the root path's
    operators), summed over the chunk and then chunk by chunk in group order.

    Parameters
    ----------
    points : Array
        ``[N, 3]``.
    tables : _GroupTables
        From :func:`_group_tables`.
    order : int
        The truncation order.
    G : float
        Gravitational constant.
    eps2_plummer : float
        The Plummer-equivalent softening squared (eq 16b).
    slice_size : int
        Points per step.
    group_chunk : int
        Groups per inner step.
    want_potential : bool
        Also the potential.
    want_scale : bool
        Also eq (16b)'s force scale.
    point_mask : Optional[Array]
        ``[N]`` active points (``None``: all).

    Returns
    -------
    tuple[Array, Array, Array]
        ``[N, 3]`` acceleration, ``[N]`` potential, ``[N]`` force scale.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.operators.real_p2m_l2p import evaluate_local_real_with_grad

    n = int(points.shape[0])
    dtype = points.dtype
    size = min(int(slice_size), n)
    steps = -(-n // size)
    k = int(tables.r_in.shape[0])
    chunk = max(1, min(int(group_chunk), k))
    nchunks = -(-k // chunk)
    pad = nchunks * chunk - k
    coeffs = (int(order) + 1) ** 2
    gconst = jnp.asarray(G, dtype)
    zero3 = jnp.zeros((3,), dtype)

    def padded(a: Array, fill: float = 0.0) -> Array:
        if not pad:
            return a
        tail = jnp.full((pad,) + tuple(a.shape[1:]), fill, a.dtype)
        return jnp.concatenate([a, tail])

    centres = padded(tables.centres).reshape(nchunks, chunk, 3)
    mults = padded(tables.multipoles[:, :coeffs]).reshape(nchunks, chunk, coeffs)
    masses = padded(tables.mass).reshape(nchunks, chunk)
    r_in2 = tables.r_in * tables.r_in

    def piece(x: Array, act: Optional[Array]) -> tuple[Array, Array, Array]:
        dall = x[:, None, :] - tables.centres[None, :, :]
        inner = jnp.any(jnp.sum(dall * dall, axis=-1) < r_in2[None, :], axis=1)
        use = ~inner if act is None else (~inner & act)

        def one_chunk(carry: tuple, xs: tuple) -> tuple[tuple, None]:
            acc, pot, scale = carry
            c, mult, mass = xs
            d = x[:, None, :] - c[None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            on = use[:, None] & (mass[None, :] > 0)
            safe = jnp.where(on[..., None], d, jnp.ones_like(d)).reshape(-1, 3)
            loc = m2l_rot_scale_real_batch(
                jnp.broadcast_to(mult[None], (size, chunk, coeffs)).reshape(-1, coeffs),
                safe,
                order=int(order),
            )
            grad, phi = jax.vmap(
                lambda cc: evaluate_local_real_with_grad(cc, zero3, order=int(order))
            )(loc)
            grad = grad.reshape(size, chunk, 3)
            phi = phi.reshape(size, chunk)
            acc = acc + jnp.sum(jnp.where(on[..., None], -gconst * grad, 0.0), axis=1)
            if want_potential:
                pot = pot + jnp.sum(jnp.where(on, -gconst * phi, 0.0), axis=1)
            if want_scale:
                f = gconst * mass[None, :] / (r2 + jnp.asarray(eps2_plummer, dtype))
                scale = scale + jnp.sum(jnp.where(on, f, 0.0), axis=1)
            return (acc, pot, scale), None

        init = (
            jnp.zeros((size, 3), dtype),
            jnp.zeros((size,), dtype),
            jnp.zeros((size,), dtype),
        )
        (acc, pot, scale), _ = jax.lax.scan(one_chunk, init, (centres, mults, masses))
        return acc, pot, scale

    def body(i: Array, carry: tuple) -> tuple:
        acc, pot, scale = carry
        start = jnp.minimum(i * size, n - size)
        x = jax.lax.dynamic_slice_in_dim(points, start, size, axis=0)
        act = (
            None
            if point_mask is None
            else jax.lax.dynamic_slice_in_dim(point_mask, start, size, axis=0)
        )
        da, dp, ds = piece(x, act)
        acc = jax.lax.dynamic_update_slice_in_dim(acc, da, start, axis=0)
        if want_potential:
            pot = jax.lax.dynamic_update_slice_in_dim(pot, dp, start, axis=0)
        if want_scale:
            scale = jax.lax.dynamic_update_slice_in_dim(scale, ds, start, axis=0)
        return acc, pot, scale

    init = (
        jnp.zeros((n, 3), dtype),
        jnp.zeros((n if want_potential else 1,), dtype),
        jnp.zeros((n if want_scale else 1,), dtype),
    )
    return jax.lax.fori_loop(0, steps, body, init)


def _split(
    provider: Any,
    topology: Any,
    *,
    mac: str,
    theta_mp: float,
    r_min: float,
    floor: float,
) -> tuple[Any, Array, Array]:
    """The source tables, the root's centre and the inner radius.

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        Its topology.
    mac : str
        The target walk's criterion.
    theta_mp : float
        The outer opening angle.
    r_min : float
        The inner radius' floor.
    floor : float
        The separation floor.

    Returns
    -------
    tuple[Any, Array, Array]
        ``(tables, centre [3], r_in)``.
    """
    src = _source_tables(
        provider.engine, topology, order=int(provider.max_order), mac=mac
    )
    root = src.root
    big_r = src.radius[root]
    dtype = src.radius.dtype
    r_in = jnp.maximum(
        jnp.maximum(big_r / jnp.asarray(theta_mp, dtype), big_r + floor),
        jnp.asarray(r_min, dtype),
    )
    return src, src.centers[root], r_in


def _check_groups(plan: BackReactionPlan, groups: Optional[SourceGroups]) -> None:
    """``groups`` must match ``plan.num_groups`` (none for a root plan).

    Parameters
    ----------
    plan : BackReactionPlan
        The plan.
    groups : Optional[SourceGroups]
        The groups given.

    Raises
    ------
    ValueError
        On a mismatch.
    """
    k = int(plan.num_groups)
    if k == 0:
        if groups is not None:
            raise ValueError("groups given to a root back-reaction plan (num_groups 0)")
        return
    if groups is None:
        raise ValueError(f"the plan has {k} source groups: pass groups=SourceGroups")
    shape = tuple(jnp.shape(groups.masses))
    if len(shape) != 2 or shape[0] != k or tuple(jnp.shape(groups.positions)) != (
        shape + (3,)
    ):
        raise ValueError(
            f"groups must be [K={k}, M, 3] positions and [K, M] masses, got "
            f"{tuple(jnp.shape(groups.positions))} / {shape}"
        )


def _inner_points(
    provider: Any,
    topology: Any,
    x: Array,
    *,
    mac: str,
    plan_like: tuple,
    floor: float,
    groups: Optional[SourceGroups],
) -> tuple[Any, Array, Any]:
    """The source tables, the inner mask and the outer description.

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        Its topology.
    x : Array
        ``[N, 3]`` points in the tables' dtype.
    mac : str
        The criterion.
    plan_like : tuple
        ``(theta_mp, r_min, mp_order)``.
    floor : float
        The separation floor.
    groups : Optional[SourceGroups]
        ``None``: the root split.

    Returns
    -------
    tuple[Any, Array, Any]
        ``(tables, inner [N] bool, outer)`` with ``outer`` ``(centre, r_in)``
        for the root split or the :class:`_GroupTables`.
    """
    theta_mp, r_min, q = plan_like
    src, centre, r_in = _split(
        provider, topology, mac=mac, theta_mp=theta_mp, r_min=r_min, floor=floor
    )
    if groups is None:
        d = x.astype(centre.dtype) - centre[None, :]
        return src, jnp.sum(d * d, axis=-1) < r_in * r_in, (centre, r_in)
    gt = _group_tables(
        groups,
        order=int(q),
        dtype=src.radius.dtype,
        theta_mp=theta_mp,
        r_min=r_min,
        floor=floor,
    )
    r_in2 = gt.r_in * gt.r_in

    def near_any(carry: Array, xs: tuple) -> tuple[Array, None]:
        c, r2 = xs
        d = x.astype(c.dtype) - c[None, :]
        return carry | (jnp.sum(d * d, axis=-1) < r2), None

    inner, _ = jax.lax.scan(
        near_any, jnp.zeros((x.shape[0],), bool), (gt.centres, r_in2)
    )
    return src, inner, gt


def evaluate_back_reaction(
    provider: Any,
    topology: Any,
    points: Array,
    *,
    plan: BackReactionPlan,
    force_scale: Optional[Array] = None,
    potential: bool = False,
    return_force_scale: bool = False,
    softening: Optional[float] = None,
    point_mask: Optional[Array] = None,
    groups: Optional[SourceGroups] = None,
) -> TargetField:
    """The field of ``topology``'s sources at ``points``: outer multipole(s) + inner walk.

    A drop-in for ``provider.target_field(topology, points, plan, ...)`` when
    the sources are compact (or K compact groups) and the points many (module
    docstring). Traced or eager.

    Parameters
    ----------
    provider : Any
        The sources' :class:`~jaccpot.runtime.fused_provider.FusedForceProvider`.
    topology : Any
        Its topology (a rebuild at the sources' positions).
    points : Array
        ``[plan.num_points, 3]``.
    plan : BackReactionPlan
        From :func:`plan_back_reaction`.
    force_scale : Optional[Array]
        ``[N]`` the points' f_b (``dehnen_error`` sources only).
    potential : bool
        Also the potential.
    return_force_scale : bool
        Also the points' f_b.
    softening : Optional[float]
        The points' pair-kernel softening, as in ``target_field``.
    point_mask : Optional[Array]
        ``[N]`` bool, the active points (``None``: all). Inactive points get
        zeros and take no part in the inner walk's groups, so they change no
        active point's field.
    groups : Optional[SourceGroups]
        The source groups of a grouped plan (``plan.num_groups = K``): the same
        particles as the topology's, by group. ``None`` for a root plan.

    Returns
    -------
    TargetField
        In the points' order and dtype; ``needs`` the inner walk's far/near
        counts, ``ok`` False when the inner points exceed the capacity or the
        inner walk overflowed, ``stats`` (queue peak, MAC tests, inner points).

    Raises
    ------
    ValueError
        If ``points`` is not ``[plan.num_points, 3]``, ``dehnen_error``
        sources have no ``force_scale``, or ``groups`` does not match the plan.
    """
    engine = provider.engine
    mac = target_mac_type(engine)
    pts = jnp.asarray(points)
    if pts.ndim != 2 or tuple(pts.shape) != (int(plan.num_points), 3):
        raise ValueError(
            f"points must be [{int(plan.num_points)}, 3] (the plan's), got "
            f"{tuple(pts.shape)}"
        )
    if mac == "dehnen_error" and force_scale is None:
        raise ValueError(
            "mac_type='dehnen_error': the inner walk needs the points' force scale"
        )
    _check_groups(plan, groups)
    eps = float(engine.softening) if softening is None else float(softening)
    act = None if point_mask is None else jnp.asarray(point_mask, bool)
    q = int(plan.mp_order)
    x = pts
    src, inner, outer = _inner_points(
        provider,
        topology,
        x,
        mac=mac,
        plan_like=(float(plan.theta_mp), float(plan.r_min), q),
        floor=_floor(provider, softening),
        groups=groups,
    )
    dtype = src.radius.dtype
    x = pts.astype(dtype)
    if groups is None:
        centre, r_in = outer
        acc, pot, scale = _outer_field(
            x,
            centre,
            r_in,
            src.multipoles[src.root][: (q + 1) ** 2],
            src.mass[src.root],
            order=q,
            G=float(engine.G),
            eps2_plummer=eps * eps,
            slice_size=int(plan.slice_size),
            want_potential=bool(potential),
            want_scale=bool(return_force_scale),
            point_mask=act,
        )
    else:
        acc, pot, scale = _outer_field_grouped(
            x,
            outer,
            order=q,
            G=float(engine.G),
            eps2_plummer=eps * eps,
            slice_size=int(plan.slice_size),
            group_chunk=int(plan.group_chunk),
            want_potential=bool(potential),
            want_scale=bool(return_force_scale),
            point_mask=act,
        )
    if act is not None:
        inner = inner & act
    count = jnp.sum(inner, dtype=jnp.int32)
    cap = plan.inner_capacity
    idx = jnp.nonzero(inner, size=cap, fill_value=0)[0].astype(jnp.int32)
    live = jnp.arange(cap, dtype=jnp.int32) < count
    fb_in = None
    if force_scale is not None:
        fb_in = jnp.where(live, jnp.asarray(force_scale, dtype)[idx], jnp.inf)
    field = _evaluate(
        provider,
        topology,
        x[idx],
        plan=plan.inner,
        force_scale=fb_in,
        potential=bool(potential),
        return_force_scale=bool(return_force_scale),
        mac=None,
        softening=softening,
        point_mask=live,
        tables=src,
    )
    # dead slots point at row 0 and carry zeros: adding them changes nothing
    acc = acc.at[idx].add(jnp.where(live[:, None], field.acc.astype(dtype), 0.0))
    if potential:
        pot = pot.at[idx].add(jnp.where(live, field.potential.astype(dtype), 0.0))
    if return_force_scale:
        scale = scale.at[idx].add(jnp.where(live, field.force_scale.astype(dtype), 0.0))
    out = pts.dtype
    stats = jnp.stack([field.stats[0], field.stats[1], count.astype(_count_dtype())])
    return TargetField(
        acc=acc.astype(out),
        potential=pot.astype(out) if potential else None,
        force_scale=scale.astype(out) if return_force_scale else None,
        needs=field.needs,
        ok=field.ok & (count <= cap),
        stats=stats,
    )


def plan_back_reaction(
    provider: Any,
    topology: Any,
    points: Array,
    *,
    theta_mp: Optional[float] = None,
    mp_order: Optional[int] = None,
    r_min: float = 0.0,
    softening: Optional[float] = None,
    headroom: float = 1.5,
    min_inner: int = 1024,
    group_size: int = 32,
    slice_size: int = 32768,
    force_scale: Optional[Array] = None,
    far: str = "m2p",
    walk_chunk: int = 2048,
    point_mask: Optional[Array] = None,
    groups: Optional[SourceGroups] = None,
    group_chunk: int = 8,
) -> BackReactionPlan:
    """Size a :class:`BackReactionPlan` eagerly at the current positions.

    Parameters
    ----------
    provider : Any
        The sources' provider.
    topology : Any
        Its topology now.
    points : Array
        ``[N, 3]`` the points now.
    theta_mp : Optional[float]
        The outer opening angle (``None``: the provider's ``theta``).
    mp_order : Optional[int]
        The outer multipoles' order (``None``: the provider's ``max_order``).
    r_min : float
        A floor on the inner radius.
    softening : Optional[float]
        The points' softening, as in :func:`evaluate_back_reaction`.
    headroom : float
        Capacity multiplier over the measured inner count and walk needs.
    min_inner : int
        The inner capacity's floor.
    group_size : int
        The inner walk's group size.
    slice_size : int
        Outer points per step.
    force_scale : Optional[Array]
        ``[N]`` the points' f_b (``dehnen_error`` sources).
    far : str
        The inner walk's far-field form (``TargetPlan.far``).
    walk_chunk : int
        The inner walk's chunk (``TargetPlan.walk_chunk``).
    point_mask : Optional[Array]
        ``[N]`` the active points (as in :func:`evaluate_back_reaction`).
    groups : Optional[SourceGroups]
        K source groups: a grouped plan (``num_groups = K``); ``None``: the root.
    group_chunk : int
        Groups per outer step (grouped plans).

    Returns
    -------
    BackReactionPlan
        A plan that fits the points now with ``headroom``.

    Raises
    ------
    ValueError
        If ``mp_order`` exceeds the provider's order.
    """
    order = int(provider.max_order)
    q = order if mp_order is None else int(mp_order)
    if q > order:
        raise ValueError(f"mp_order {q} exceeds the provider's max_order {order}")
    th = float(provider.theta_value) if theta_mp is None else float(theta_mp)
    pts = jnp.asarray(points)
    floor = _floor(provider, softening)
    mac = target_mac_type(provider.engine)
    act = None if point_mask is None else jnp.asarray(point_mask, bool)

    @jax.jit
    def inner_mask(topo: Any, x: Array, grp: Optional[SourceGroups]) -> Array:
        _, inner, _ = _inner_points(
            provider,
            topo,
            x,
            mac=mac,
            plan_like=(th, float(r_min), q),
            floor=floor,
            groups=grp,
        )
        return inner

    mask = inner_mask(topology, pts, groups)
    if act is not None:
        mask = mask & act
    mask = np.asarray(mask)
    count = int(mask.sum())
    cap = _ceil_to(max(count * float(headroom), int(min_inner)), int(group_size))
    cap = min(cap, _ceil_to(int(pts.shape[0]), int(group_size)))
    idx = np.nonzero(mask)[0]
    sel = np.zeros((cap,), np.int64)
    sel[: min(count, cap)] = idx[:cap]
    live = jnp.asarray(np.arange(cap) < count)
    fb = None
    if force_scale is not None:
        fb = jnp.where(live, jnp.asarray(force_scale)[jnp.asarray(sel)], jnp.inf)
    inner = plan_target_points(
        provider,
        topology,
        pts[jnp.asarray(sel)],
        group_size=int(group_size),
        headroom=float(headroom),
        force_scale=fb,
        softening=softening,
        point_mask=live,
        far=str(far),
        walk_chunk=int(walk_chunk),
    )
    return BackReactionPlan(
        num_points=int(pts.shape[0]),
        inner=inner,
        theta_mp=th,
        mp_order=q,
        r_min=float(r_min),
        slice_size=int(min(int(slice_size), int(pts.shape[0]))),
        num_groups=0 if groups is None else int(jnp.shape(groups.masses)[0]),
        group_chunk=int(group_chunk),
    )
