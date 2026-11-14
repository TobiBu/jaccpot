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
from typing import Any, Optional

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
    """

    num_points: int
    inner: TargetPlan
    theta_mp: float
    mp_order: int
    r_min: float = 0.0
    slice_size: int = 32768

    def __post_init__(self) -> None:
        if int(self.num_points) < 1 or int(self.slice_size) < 1:
            raise ValueError("BackReactionPlan: num_points and slice_size >= 1")
        if int(self.mp_order) < 0:
            raise ValueError("BackReactionPlan.mp_order must be >= 0")
        if not 0.0 < float(self.theta_mp) < 1.0:
            raise ValueError("BackReactionPlan.theta_mp must be in (0, 1)")
        if float(self.r_min) < 0.0:
            raise ValueError("BackReactionPlan.r_min must be >= 0")

    @property
    def inner_capacity(self) -> int:
        """Inner points the plan holds."""
        return int(self.inner.num_points)


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

    def piece(x: Array) -> tuple[Array, Array, Array]:
        d = x - centre[None, :]
        r2 = jnp.sum(d * d, axis=-1)
        outer = r2 >= r_in * r_in
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
        da, dp, ds = piece(x)
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
) -> TargetField:
    """The field of ``topology``'s sources at ``points``: outer multipole + inner walk.

    A drop-in for ``provider.target_field(topology, points, plan, ...)`` when
    the sources are compact and the points many (module docstring). Traced or
    eager.

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

    Returns
    -------
    TargetField
        In the points' order and dtype; ``needs`` the inner walk's far/near
        counts, ``ok`` False when the inner points exceed the capacity or the
        inner walk overflowed, ``stats`` (queue peak, MAC tests, inner points).

    Raises
    ------
    ValueError
        If ``points`` is not ``[plan.num_points, 3]``, or ``dehnen_error``
        sources have no ``force_scale``.
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
    eps = float(engine.softening) if softening is None else float(softening)
    src, centre, r_in = _split(
        provider,
        topology,
        mac=mac,
        theta_mp=float(plan.theta_mp),
        r_min=float(plan.r_min),
        floor=_floor(provider, softening),
    )
    dtype = src.radius.dtype
    x = pts.astype(dtype)
    q = int(plan.mp_order)
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
    )
    d = x - centre[None, :]
    inner = jnp.sum(d * d, axis=-1) < r_in * r_in
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
        scale = scale.at[idx].add(
            jnp.where(live, field.force_scale.astype(dtype), 0.0)
        )
    out = pts.dtype
    stats = jnp.stack(
        [field.stats[0], field.stats[1], count.astype(_count_dtype())]
    )
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
        The outer multipole's order (``None``: the provider's ``max_order``).
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

    @jax.jit
    def inner_mask(topo: Any, x: Array) -> tuple[Array, Array]:
        _, centre, r_in = _split(
            provider, topo, mac=mac, theta_mp=th, r_min=float(r_min), floor=floor
        )
        d = x.astype(centre.dtype) - centre[None, :]
        return jnp.sum(d * d, axis=-1) < r_in * r_in, r_in

    mask, _ = inner_mask(topology, pts)
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
    )
