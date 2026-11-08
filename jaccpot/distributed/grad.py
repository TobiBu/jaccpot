"""Reverse-mode gradients of the fused multi-GPU force: the plumbing (WP-G, G1).

The single card differentiates its fused force by rebuilding the topology from the
inputs' STOPPED gradients and re-evaluating at the live positions and masses
through the production kernels' ``custom_vjp`` seams (S8/S9,
``FusedForceProvider.differentiable_step``). On the mesh the same pattern runs on
every device inside ``shard_map`` (:func:`mesh_differentiable_force_shard`):

1. the template route's rebuild (``template_force._shard_rebuild``: the global box,
   the provider's rebuild over the live rows, the cross hook between the sweeps) on
   ``stop_gradient(x, m)``, the walk's far list kept, and the hook built with a
   ``grad_sink`` (:func:`make_mesh_grad_hook`) that records the frozen cross layout
   of this trace -- what was sent where, what arrived, which pairs -- into a
   :class:`CrossGradPlan`;
2. the gradient plan of the local lists (template-derived statics,
   :func:`mesh_grad_statics`; the device symmetry verdict) and the CROSS verdict
   (no cross overflow, and every device's per-peer pair hashes equal at both ends of
   each pair, :func:`cross_verdict`);
3. S8's differentiable evaluation at the live ``x, m`` with the cross far sources
   in its far chain (:class:`~jaccpot.runtime._large_n_grad.LargeNCrossFar`: the
   imports stopped, merged into the local CSR rows exactly as the production
   refresh merges them), the M2L on the merged CSR through **seam A**;
4. the cross NEAR term at the live sorted particles through **seam B**, added in
   input row order as production adds it; padding rows zeroed; the gradient gate.

The exchanges are never transposed: the seams' reverses send OUTPUT cotangents
forward along the frozen layout (design §2). The defaults are
:func:`cross_far_seam` (seam A,
:func:`~jaccpot.pallas.m2l_real_csr_lanes_cross_vjp.cross_far_m2l_cvjp`) and
:func:`cross_near_seam` (seam B,
:func:`~jaccpot.distributed.cross_vjp.cross_near_acceleration_direct_cvjp`); the
forward is ``FusedMeshForce.shard``'s, bitwise. :func:`placeholder_far_seam` and
:func:`placeholder_near_seam` (the production launches forward, a reverse that
raises) stay for tests of the plumbing alone. :func:`make_mesh_grad_fn` wraps the
per-device force in the mesh's ``shard_map`` for a loss outside it.

Seam contracts:

* ``far_seam(plan: CrossGradPlan) -> m2l``: ``m2l`` is called in place of
  :func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`, with
  its 13 arguments positionally (``multipoles [n_local + K, C], centers, sources,
  row_offsets, active_pair_count, order, k_lanes, interpret, backend, num_warps,
  n_targets = n_local, presorted = True, symmetric = False``) on the merged CSR, and
  returns ``[n_local, C]`` local increments;
* ``near_seam(plan, positions_sorted, masses_sorted, *, leaf_ranges,
  particle_indices, num_internal, leaf_width, softening_sq, G, softening_kernel,
  interpret, with_force_scale) -> acc | (acc, f_b share)``: the cross near term of
  :func:`~jaccpot.distributed.cross.cross_near_acceleration_direct` in INPUT row
  order, differentiable in the LOCAL sorted particles (and ``softening_sq``, ``G``);
  the imports are the plan's.

Refused (``NotImplementedError``): forward mode (``ad_mode="forward"``), every
control exchange and the table near layout (``cross.make_cross_hook``'s grad-sink
refusals), and statics from a template that changed since they were derived.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    "CrossGradPlan",
    "cross_verdict",
    "make_mesh_grad_fn",
    "make_mesh_grad_hook",
    "mesh_differentiable_force_shard",
    "mesh_grad_statics",
    "cross_far_seam",
    "cross_near_seam",
    "placeholder_far_seam",
    "placeholder_near_seam",
]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "send_node",
        "alive",
        "send_sizes",
        "recv_sizes",
        "recv_multipoles",
        "recv_centers",
        "far_sources",
        "far_targets",
        "src",
        "live_q",
        "part_sizes",
        "recv_particles",
        "recv_psizes",
        "row_start",
        "row_count",
        "near_targets",
        "near_sources",
        "near_count",
        "far_overflow",
        "near_overflow",
        "pair_hashes",
        "cross_masses",
    ],
    meta_fields=[
        "ndev",
        "axis_name",
        "n_local",
        "n_coeff",
        "leaf_width",
        "send_node_cap",
        "recv_node_cap",
        "send_particle_cap",
        "recv_particle_cap",
        "error_order",
    ],
)
@dataclass(frozen=True)
class CrossGradPlan:
    """One device's frozen cross-domain layout, as its rebuild's hook recorded it.

    Made inside the trace that ran the hook (:meth:`from_sink`), from
    ``make_cross_hook(grad_sink=...)``'s record of the default (symmetric)
    exchange. A pytree: the integer layout and the stopped imports are its leaves,
    the capacities and the mesh axis static. The reverse seams replay the forward's
    two ragged exchanges on it (send the cotangents of the rows a device imported
    back to their owners ALONG the forward layout), so nothing is sorted,
    scattered by source or transposed.

    Attributes
    ----------
    send_node : Array
        ``(send_node_cap,)`` the local node behind every far send row, rows
        grouped by peer (``exp_f.rows``' nodes; dead rows read node 0).
    alive : Array
        ``(send_node_cap,)`` live far send rows.
    send_sizes : Array
        ``(ndev,)`` far rows sent to each peer (the exchange's send sizes).
    recv_sizes : Array
        ``(ndev,)`` far rows received from each peer (predicted locally; the
        exchange's sizes are checked against them into ``far_overflow``).
    recv_multipoles : Array
        ``(recv_node_cap, C)`` the imported multipoles (stopped).
    recv_centers : Array
        ``(recv_node_cap, 3)`` their expansion centres (stopped).
    far_sources : Array
        ``(P_f,)`` cross far pair sources, ``n_local``-offset rows of the import,
        ``-1``-padded, (target, source) order.
    far_targets : Array
        ``(P_f,)`` their local target nodes, ``-1``-padded.
    src : Array
        ``(send_particle_cap,)`` the sorted particle row behind every near send
        slot.
    live_q : Array
        ``(send_particle_cap,)`` live near send slots.
    part_sizes : Array
        ``(ndev,)`` particles sent to each peer.
    recv_particles : Array
        ``(recv_particle_cap, 4)`` the imported ``[x, m]`` (stopped).
    recv_psizes : Array
        ``(ndev,)`` particles received from each peer (the exchange's sizes).
    row_start, row_count : Array
        ``(recv_node_cap,)`` each received near row's range in ``recv_particles``.
    near_targets : Array
        ``(P_n,)`` cross near pairs' local target LEAF nodes, ``-1``-padded.
    near_sources : Array
        ``(P_n,)`` their received rows, ``-1``-padded.
    near_count : Array
        The near walk's pair count (the near term's live prefix).
    far_overflow, near_overflow : Array
        The hook's far and near overflow flags (summary included).
    pair_hashes : Array
        ``uint64[ndev, 2, 2]``: per peer, far then near, the two order-free hash
        sums of this device's cross pairs with it (``cross._peer_pair_hashes``).
    cross_masses : Optional[Array]
        ``mac_type='dehnen_error'``: the imported rows' masses (stopped), else
        ``None``.
    ndev : int
        Mesh size.
    axis_name : str
        Mesh axis.
    n_local : int
        Local node count (the imported sources sit at ``n_local + row``).
    n_coeff : int
        Packed multipole coefficients.
    leaf_width : int
        The cross capacities' leaf width (the near term's tile width floor).
    send_node_cap, recv_node_cap, send_particle_cap, recv_particle_cap : int
        The exchange buffers' static capacities.
    error_order : int
        eq (16a)'s order under the criterion, else 0.
    """

    send_node: Array
    alive: Array
    send_sizes: Array
    recv_sizes: Array
    recv_multipoles: Array
    recv_centers: Array
    far_sources: Array
    far_targets: Array
    src: Array
    live_q: Array
    part_sizes: Array
    recv_particles: Array
    recv_psizes: Array
    row_start: Array
    row_count: Array
    near_targets: Array
    near_sources: Array
    near_count: Array
    far_overflow: Array
    near_overflow: Array
    pair_hashes: Array
    cross_masses: Optional[Array]
    ndev: int
    axis_name: str
    n_local: int
    n_coeff: int
    leaf_width: int
    send_node_cap: int
    recv_node_cap: int
    send_particle_cap: int
    recv_particle_cap: int
    error_order: int

    @classmethod
    def from_sink(cls, sink: dict) -> "CrossGradPlan":
        """The plan from a hook's ``grad_sink``, in the trace that filled it.

        Parameters
        ----------
        sink : dict
            ``make_cross_hook(grad_sink=sink)``'s record of this trace.

        Returns
        -------
        CrossGradPlan
            The layout, the imports with their gradients stopped.

        Raises
        ------
        RuntimeError
            If the sink holds no record (the hook did not run in this trace, or
            ran without its near half).
        """
        from jaccpot.distributed.cross import (
            GRAD_SINK_FAR_FIELDS,
            GRAD_SINK_NEAR_FIELDS,
        )

        missing = [
            k
            for k in GRAD_SINK_FAR_FIELDS + GRAD_SINK_NEAR_FIELDS + ("far_overflow",)
            if k not in sink
        ]
        if missing:
            raise RuntimeError(
                "the cross hook's grad_sink holds no record of this trace (missing "
                f"{missing}): the hook must run, with its near sink, in the same "
                "trace that builds the plan"
            )
        stop = jax.lax.stop_gradient
        caps = sink["caps"]
        mp, cen, src_f, tgt_f = sink["cross_far"]
        masses = sink.get("cross_masses")
        return cls(
            send_node=sink["send_node"],
            alive=sink["alive"],
            send_sizes=sink["send_sizes"],
            recv_sizes=sink["recv_sizes"],
            recv_multipoles=stop(mp),
            recv_centers=stop(cen),
            far_sources=src_f,
            far_targets=tgt_f,
            src=sink["src"],
            live_q=sink["live_q"],
            part_sizes=sink["part_sizes"],
            recv_particles=stop(sink["recv_p"]),
            recv_psizes=sink["recv_psizes"],
            row_start=sink["row_start"],
            row_count=sink["row_count"],
            near_targets=sink["target_node"],
            near_sources=sink["source_row"],
            near_count=sink["count"],
            far_overflow=jnp.asarray(sink["far_overflow"], jnp.bool_),
            near_overflow=jnp.asarray(sink["near_overflow"], jnp.bool_),
            pair_hashes=jnp.stack([sink["far_hashes"], sink["near_hashes"]], axis=1),
            cross_masses=None if masses is None else stop(masses),
            ndev=int(sink["ndev"]),
            axis_name=str(sink["axis_name"]),
            n_local=int(sink["n_local"]),
            n_coeff=int(sink["n_coeff"]),
            leaf_width=int(sink["leaf_width"]),
            send_node_cap=int(caps.send_node_cap),
            recv_node_cap=int(caps.recv_node_cap),
            send_particle_cap=int(caps.send_particle_cap),
            recv_particle_cap=int(caps.recv_particle_cap),
            error_order=int(sink.get("error_order", 0)),
        )

    def cross_far(self, reverse: Optional[Callable[..., Array]] = None) -> Any:
        """The far chain's cross sources, as the production refresh passes them.

        Parameters
        ----------
        reverse : Optional[Callable[..., Array]]
            The bound M2L seam (``far_seam(plan)``), or ``None`` for the
            production (forward-only) one.

        Returns
        -------
        LargeNCrossFar
            ``(multipoles, centers, sources, targets, reverse)``.
        """
        from jaccpot.runtime._large_n_grad import LargeNCrossFar

        return LargeNCrossFar(
            self.recv_multipoles,
            self.recv_centers,
            self.far_sources,
            self.far_targets,
            reverse,
        )

    def near_sink(self) -> dict:
        """The near sink the production cross near term reads, from the plan.

        Returns
        -------
        dict
            ``particles``, ``row_start``, ``row_count``, ``target_node``,
            ``source_row``, ``count`` (no ``leaf_width``: the caller passes the
            tile width).
        """
        return {
            "particles": self.recv_particles,
            "row_start": self.row_start,
            "row_count": self.row_count,
            "target_node": self.near_targets,
            "source_row": self.near_sources,
            "count": self.near_count,
        }


def cross_verdict(plan: CrossGradPlan) -> tuple[Array, Array]:
    """Whether the cross lists are ones the reverse seams can replay.

    One ``all_gather`` of every device's per-peer pair hashes: the ``[ndev, ndev]``
    matrix must be symmetric (each pair seen alike from both ends -- the symmetric
    walk's contract, G0.1), and no cross buffer may have overflowed.

    Parameters
    ----------
    plan : CrossGradPlan
        This device's plan (inside the ``shard_map``).

    Returns
    -------
    tuple[Array, Array]
        ``(ok, symmetric)`` boolean scalars, ``ok = symmetric & no overflow``; the
        same ``symmetric`` on every device.
    """
    hashes = jax.lax.all_gather(plan.pair_hashes, plan.axis_name, tiled=False)
    symmetric = jnp.all(hashes == jnp.swapaxes(hashes, 0, 1))
    ok = symmetric & ~plan.far_overflow & ~plan.near_overflow
    return ok, symmetric


# --------------------------------------------------------------------------
# The seams' placeholders: the production launches forward, no reverse yet.
# --------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def _far_placeholder(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Optional[Array],
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_targets: Optional[int] = None,
    presorted: bool = False,
    symmetric: bool = False,
) -> Array:
    """Seam A's placeholder: the production CSR M2L launch, no reverse.

    Parameters
    ----------
    multipoles, centers, sources, targets, active_pair_count : Array
        As :func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`.
    order, k_lanes, interpret, backend, num_warps, n_targets, presorted, symmetric
        Its static arguments (``symmetric`` unused forward).

    Returns
    -------
    Array
        ``[n_targets, C]`` local increments, bitwise the production launch's.
    """
    del symmetric
    from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas

    return m2l_real_csr_lanes_pallas(  # pyright: ignore[reportReturnType]
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=n_targets,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=presorted,
    )


def _far_placeholder_fwd(*args: Any) -> tuple[Array, None]:
    return _far_placeholder(*args), None


def _far_placeholder_bwd(*args: Any) -> Any:
    raise NotImplementedError("cross seam A not wired yet")


_far_placeholder.defvjp(_far_placeholder_fwd, _far_placeholder_bwd)


def placeholder_far_seam(plan: Optional[CrossGradPlan]) -> Callable[..., Array]:
    """Seam A's default: the production M2L launch forward, a reverse that raises.

    Parameters
    ----------
    plan : Optional[CrossGradPlan]
        The device's plan (unused: the real seam binds its far layout).

    Returns
    -------
    Callable[..., Array]
        A ``custom_vjp`` with ``m2l_real_csr_lanes_pallas_cvjp``'s signature whose
        backward raises ``NotImplementedError("cross seam A not wired yet")``.
    """
    del plan
    return _far_placeholder  # pyright: ignore[reportReturnType]


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _near_placeholder(
    near: dict,
    positions_sorted: Array,
    masses_sorted: Array,
    topo: tuple,
    softening_sq: Array,
    G: Array,
    num_internal: int,
    leaf_width: int,
    softening_kernel: Optional[str],
    interpret: bool,
    with_force_scale: bool,
) -> Any:
    """Seam B's placeholder: the production cross near launch, no reverse.

    Parameters
    ----------
    near : dict
        :meth:`CrossGradPlan.near_sink`.
    positions_sorted, masses_sorted : Array
        This device's particles in tree order.
    topo : tuple
        ``(leaf_ranges, particle_indices)``.
    softening_sq, G : Array
        The pair kernel's scalars.
    num_internal, leaf_width, softening_kernel, interpret, with_force_scale
        Static, as :func:`~jaccpot.distributed.cross.cross_near_acceleration_direct`.

    Returns
    -------
    Any
        Its result, bitwise.
    """
    from jaccpot.distributed.cross import cross_near_acceleration_direct

    leaf_ranges, particle_indices = topo
    return cross_near_acceleration_direct(
        near,
        positions_sorted,
        masses_sorted,
        leaf_ranges,
        particle_indices,
        num_internal,
        leaf_width=leaf_width,
        softening_sq=softening_sq,
        G=G,
        interpret=interpret,
        softening_kernel=softening_kernel,
        with_force_scale=with_force_scale,
    )


def _near_placeholder_fwd(*args: Any) -> tuple[Any, None]:
    return _near_placeholder(*args), None


def _near_placeholder_bwd(*args: Any) -> Any:
    raise NotImplementedError("cross seam B not wired yet")


_near_placeholder.defvjp(_near_placeholder_fwd, _near_placeholder_bwd)


def placeholder_near_seam(
    plan: CrossGradPlan,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    leaf_ranges: Array,
    particle_indices: Array,
    num_internal: int,
    leaf_width: int,
    softening_sq: Array,
    G: Array,
    softening_kernel: Optional[str],
    interpret: bool,
    with_force_scale: bool = False,
) -> Any:
    """Seam B's default: the production cross near term forward, a reverse that raises.

    Parameters
    ----------
    plan : CrossGradPlan
        The device's plan (the near imports and pairs).
    positions_sorted : Array
        ``(n, 3)`` this device's LIVE particles in tree order.
    masses_sorted : Array
        ``(n,)`` their masses.
    leaf_ranges : Array
        ``(L, 2)`` every local leaf's inclusive particle range.
    particle_indices : Array
        ``(n,)`` sorted row -> input row.
    num_internal : int
        Internal node count of the local tree.
    leaf_width : int
        The target tiles' width (the larger of the local leaves' and the plan's).
    softening_sq, G : Array
        The pair kernel's scalars, in the local force's dtype.
    softening_kernel : Optional[str]
        The pair kernel.
    interpret : bool
        Pallas interpret mode.
    with_force_scale : bool
        Also return the imported leaves' share of eq (16b)'s ``f_b``.

    Returns
    -------
    Any
        ``(n, 3)`` in input row order (and the ``(n,)`` share); its cotangent
        raises ``NotImplementedError("cross seam B not wired yet")``.
    """
    return _near_placeholder(
        plan.near_sink(),
        positions_sorted,
        masses_sorted,
        (leaf_ranges, particle_indices),
        softening_sq,
        G,
        int(num_internal),
        int(leaf_width),
        softening_kernel,
        bool(interpret),
        bool(with_force_scale),
    )


# --------------------------------------------------------------------------
# The real seams (G4): G2's cross M2L reverse and G3's cross near reverse, bound
# to the plan's frozen layout.
# --------------------------------------------------------------------------


def cross_far_seam(plan: CrossGradPlan) -> Callable[..., Array]:
    """Seam A: the merged M2L launch whose reverse completes the cross cotangents.

    Binds :func:`~jaccpot.pallas.m2l_real_csr_lanes_cross_vjp.cross_far_m2l_cvjp`
    to the plan's far layout (``send_node``, ``alive``, ``send_sizes``, the import
    capacity and the mesh axis): the returned callable takes
    ``m2l_real_csr_lanes_pallas_cvjp``'s 13 positional arguments, as
    ``_downward_prep`` calls it on the merged ``[local ; imported]`` CSR. Its
    forward is the production launch, bitwise; its reverse sends the local
    cotangents of the exported nodes forward along the far exchange and runs the
    symmetric sort-free reverse over the local rows (exact on a symmetric merged
    list: the caller's gate).

    Parameters
    ----------
    plan : CrossGradPlan
        The device's frozen cross layout.

    Returns
    -------
    Callable[..., Array]
        The bound seam.
    """
    from jaccpot.pallas.m2l_real_csr_lanes_cross_vjp import cross_far_m2l_cvjp

    def m2l(
        multipoles: Array,
        centers: Array,
        sources: Array,
        row_offsets: Array,
        active_pair_count: Optional[Array],
        order: int,
        k_lanes: int,
        interpret: bool,
        backend: str,
        num_warps: int,
        n_targets: Optional[int] = None,
        presorted: bool = False,
        symmetric: bool = False,
    ) -> Array:
        # the merged list: `_l2l` clears `symmetric` once cross pairs are in; the
        # local half's symmetry is the gradient plan's verdict (gated)
        del symmetric
        if n_targets is None or int(n_targets) != plan.n_local or not presorted:
            raise ValueError(
                "cross_far_seam: the merged launch must target the plan's "
                f"{plan.n_local} local rows on a target-sorted CSR; got "
                f"n_targets={n_targets}, presorted={presorted}"
            )
        return cross_far_m2l_cvjp(
            multipoles,
            centers,
            sources,
            row_offsets,
            active_pair_count,
            plan.send_node,
            plan.alive,
            plan.send_sizes,
            int(order),
            int(k_lanes),
            bool(interpret),
            str(backend),
            int(num_warps),
            int(plan.n_local),
            int(plan.recv_node_cap),
            str(plan.axis_name),
        )

    return m2l


def cross_near_seam(
    plan: CrossGradPlan,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    leaf_ranges: Array,
    particle_indices: Array,
    num_internal: int,
    leaf_width: int,
    softening_sq: Array,
    G: Array,
    softening_kernel: Optional[str],
    interpret: bool,
    with_force_scale: bool = False,
) -> Any:
    """Seam B: the cross near term whose reverse completes the imported sinks' cotangents.

    :func:`~jaccpot.distributed.cross_vjp.cross_near_acceleration_direct_cvjp` on
    the plan's near imports and pairs; arguments as :func:`placeholder_near_seam`.
    Forward: the production term, bitwise. Reverse: the cotangents of the sinks a
    peer imported travel forward along the near exchange, then the direct
    reverse over the local target leaves (no self term); ``softening_sq`` and
    ``G`` get this device's partial cotangents (the ``shard_map`` transpose sums
    replicated inputs' cotangents over the mesh).

    Returns
    -------
    Any
        ``(n, 3)`` in input row order (and the ``(n,)`` ``f_b`` share, whose
        cotangent is dropped).
    """
    from jaccpot.distributed.cross_vjp import cross_near_acceleration_direct_cvjp

    return cross_near_acceleration_direct_cvjp(
        plan.near_sink(),
        positions_sorted,
        masses_sorted,
        leaf_ranges,
        particle_indices,
        int(num_internal),
        src=plan.src,
        live_q=plan.live_q,
        part_sizes=plan.part_sizes,
        axis_name=str(plan.axis_name),
        leaf_width=int(leaf_width),
        softening_sq=softening_sq,
        G=G,
        interpret=bool(interpret),
        softening_kernel=softening_kernel,
        with_force_scale=bool(with_force_scale),
    )


# --------------------------------------------------------------------------
# The hook, the statics, the per-device force.
# --------------------------------------------------------------------------


def make_mesh_grad_hook(
    solver: Any,
    mesh: Any,
    *,
    theta: float,
    cross_caps: Any = None,
    near_theta: Optional[float] = None,
    axis_name: str,
) -> tuple[Optional[Callable[..., Any]], Optional[dict], Optional[dict]]:
    """The mesh's cross hook with a ``grad_sink``, as ``make_mesh_cross_hook`` builds it.

    The same hook (capacities, separation floor, criterion) plus the reverse-mode
    record; its forward is the production hook's.

    Parameters
    ----------
    solver : Any
        The solver.
    mesh : Any
        The 1-D mesh.
    theta : float
        Opening angle of the cross walks.
    cross_caps : Any
        The force's :class:`~jaccpot.distributed.cross.CrossCapacities`.
    near_theta : Optional[float]
        Passed to the hook (anything but ``None`` / ``theta`` is refused).
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Optional[Callable[..., Any]], Optional[dict], Optional[dict]]
        ``(hook, near_sink, grad_sink)``; ``(None, None, None)`` on one device.
    """
    from jaccpot.distributed.cross import make_cross_hook
    from jaccpot.distributed.rollout import _mesh_devices

    ndev = len(_mesh_devices(mesh))
    if ndev < 2:
        return None, None, None
    engine = getattr(solver, "_impl", solver)
    near: dict = {}
    sink: dict = {}
    hook = make_cross_hook(
        ndev=ndev,
        theta=theta,
        caps=cross_caps,
        near_sink=near,
        near_theta=near_theta,
        axis_name=axis_name,
        separation_floor=engine._walk_separation_floor(),
        error_criterion=bool(engine._flat_walk_criterion_active()),
        grad_sink=sink,
    )
    return hook, near, sink


def mesh_grad_statics(
    force: Any, positions: Array, masses: Array, counts: Array
) -> Any:
    """The gradient plan's static fields, derived from the mesh's TEMPLATE (G0 option a).

    ``jax.eval_shape`` of a ``keep_far_list`` rebuild under the mesh's
    ``shard_map`` (no eager prepare of a full state, nothing runs): the order, the
    leaf capacity, the far list's capacity and type from the traced topology, the
    centre mode and chunks from the runtime overrides. The far list is the
    sort-free M2L reverse's (``far_pairs_symmetric``) exactly when it is the GPU
    walk's target-sorted CSR (symmetric by construction, checked on the device
    every step); the CPU flat walk's list is neither (G0.3).

    Parameters
    ----------
    force : Any
        A prepared :class:`~jaccpot.distributed.template_force.FusedMeshForce`.
    positions : Array
        ``(ndev * cap, 3)`` mesh-sharded positions (shapes only are read).
    masses : Array
        ``(ndev * cap,)`` masses.
    counts : Array
        ``(ndev,)`` live rows per device.

    Returns
    -------
    LargeNGradStatics
        The frozen statics.
    """
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.fused import global_mesh_bounds
    from jaccpot.runtime._interaction_cache import TargetSortedFarPairs
    from jaccpot.runtime._large_n_grad import LargeNGradStatics

    setup = force._setup_or_raise()
    provider = setup.provider
    engine = provider.engine
    axis = force.axis_name
    holder: dict = {}

    def probe(x: Array, m: Array, n: Array) -> Array:
        live = jnp.minimum(n[0], int(x.shape[0]))
        bounds = global_mesh_bounds(
            x,
            num_valid=live,
            axis_name=axis,
            cubic=bool(getattr(engine, "_tree_cubic_bounds", False)),
        )
        scale = (
            {"force_scale": jnp.zeros(x.shape[:1], x.dtype)}
            if provider.scale_carry
            else {}
        )
        topo = provider.rebuild(
            x, m, bounds=bounds, num_valid=live, keep_far_list=True, **scale
        )
        st = topo.prepared
        far = st.compact_far_pairs
        holder["fields"] = (
            int(st.local_order),
            int(st.max_leaf_size),
            int(jnp.asarray(far.sources).shape[0]),
            isinstance(far, TargetSortedFarPairs),
            int(st.positions_sorted.shape[0]),
        )
        return jnp.zeros((1,), jnp.int32)

    spec = P(axis)
    with force.context():
        force.before_trace()
        try:
            jax.eval_shape(
                jax.shard_map(
                    probe,
                    mesh=force.mesh,
                    in_specs=(spec,) * 3,
                    out_specs=spec,
                    check_vma=False,
                ),
                positions,
                masses,
                counts,
            )
        finally:
            # the probe's traced walk recorded its caps: no later trace reads them
            force.before_trace()
            engine._last_refresh_capacity_ok = None
            engine._last_refresh_walk_needs = None
            engine._last_refresh_tree_overflow = None
            engine._last_refresh_tree_needs = None
    order, leaf_cap, num_far, target_sorted, num_sorted = holder["fields"]
    overrides = engine._resolve_runtime_execution_overrides(num_particles=num_sorted)
    return LargeNGradStatics(
        order=order,
        max_leaf_size=leaf_cap,
        center_mode=str(overrides.center_mode),
        m2l_chunk_size=overrides.m2l_chunk_size,
        l2l_chunk_size=overrides.l2l_chunk_size,
        num_far_pairs=num_far,
        far_pairs_target_sorted=bool(target_sorted),
        far_pairs_symmetric=bool(target_sorted),
    )


def _near_scalars(engine: Any, force: Any, dtype: Any) -> tuple[Array, Array]:
    """The cross near term's ``(softening_sq, G)``, as production builds them.

    Parameters
    ----------
    engine : Any
        The runtime engine.
    force : Any
        The resolved :class:`~jaccpot.runtime.fused_topology.ForceParams`, or
        ``None`` for the engine's own (``fused._cross_near_lanes``' expressions).
    dtype : Any
        The local force's dtype.

    Returns
    -------
    tuple[Array, Array]
        ``(softening_sq, G)``.
    """
    if force is None:
        return (
            jnp.asarray(engine.softening, dtype) ** 2,
            jnp.asarray(engine.G, dtype),
        )
    return jnp.asarray(force.eps2, dtype), jnp.asarray(force.G, dtype)


def mesh_differentiable_force_shard(
    setup: Any,
    positions: Array,
    masses: Array,
    live: Array,
    *,
    statics: Any,
    cross_hook: Optional[Callable[..., Any]],
    cross_near_sink: Optional[dict],
    grad_sink: Optional[dict],
    axis_name: str,
    force_scale: Optional[Array] = None,
    params: Any = None,
    far_seam: Callable[[CrossGradPlan], Callable[..., Array]] = cross_far_seam,
    near_seam: Callable[..., Any] = cross_near_seam,
) -> tuple[Array, ...]:
    """One device's self-gravity, differentiable in ``positions`` and ``masses``.

    Inside ``shard_map`` (``check_vma=False``), like
    :func:`~jaccpot.distributed.template_force.mesh_force_shard`, whose forward it
    is bitwise (MG1). The topology is rebuilt from the inputs' stopped gradients
    with the grad hook (:func:`make_mesh_grad_hook`) and the far list kept; the
    force is S8's differentiable evaluation at the live inputs with the cross far
    sources in its far chain (seam A) plus the cross near term (seam B), padding
    rows zeroed. The gradient is NaN unless every device's local lists are
    symmetric, the cross pair hashes agree at both ends, and nothing overflowed
    (:func:`~jaccpot.runtime.fused_provider._gradient_gate`, a ``where``); a list
    failure sets ``OVERFLOW_GRAD_LISTS`` in this device's bits.

    Parameters
    ----------
    setup : Any
        The force's ``MeshForceSetup``.
    positions : Array
        ``(cap, 3)`` this device's positions (differentiated).
    masses : Array
        ``(cap,)`` masses, zero on padding (differentiated).
    live : Array
        Scalar live row count.
    statics : Any
        :func:`mesh_grad_statics`' ``LargeNGradStatics``.
    cross_hook, cross_near_sink, grad_sink : Optional
        :func:`make_mesh_grad_hook`'s (all ``None`` on one device).
    axis_name : str
        Mesh axis.
    force_scale : Optional[Array]
        ``(cap,)`` the previous ``f_b`` under ``mac_type='dehnen_error'``
        (stopped), else ``None``.
    params : Any
        ``ForceParams`` / ``StepParams`` for the local evaluation and the cross
        near term (G4's MG9); ``None``: the engine's.
    far_seam : Callable
        Seam A's factory (module docstring); default :func:`cross_far_seam`.
    near_seam : Callable
        Seam B (module docstring); default :func:`cross_near_seam`.

    Returns
    -------
    tuple[Array, ...]
        ``(acceleration, bits, needs)`` as ``mesh_force_shard`` returns them (bits
        with ``OVERFLOW_GRAD_LISTS`` when the lists are not ones the reverses
        read), then ``f_b`` (stopped) with ``force_scale``.

    Raises
    ------
    NotImplementedError
        In forward mode: the mesh's tangent seams are not wired.
    """
    from jaccpot.distributed.template_force import _shard_rebuild, _shard_verdict
    from jaccpot.runtime._large_n_grad import traced_large_n_grad_plan
    from jaccpot.runtime.fused_provider import _gradient_gate
    from jaccpot.runtime.fused_topology import OVERFLOW_GRAD_LISTS, overflow_ok
    from jaccpot.runtime.grad_options import forward_mode_enabled

    if forward_mode_enabled():
        raise NotImplementedError(
            "the multi-GPU fused force has no forward-mode (ad_mode='forward') "
            "seams yet; differentiate it in reverse mode"
        )
    provider = setup.provider
    engine = provider.engine
    stop = jax.lax.stop_gradient
    topology = _shard_rebuild(
        provider,
        stop(positions),
        stop(masses),
        live,
        cross_hook=cross_hook,
        cross_near_sink=cross_near_sink,
        axis_name=axis_name,
        force_scale=None if force_scale is None else stop(force_scale),
        keep_far_list=True,
    )
    prepared = topology.prepared
    plan, local_symmetric = traced_large_n_grad_plan(engine, prepared, statics)
    cplan = None
    cross = None
    cross_ok = jnp.asarray(True)
    if cross_hook is not None:
        assert grad_sink is not None
        cplan = CrossGradPlan.from_sink(grad_sink)
        cross = cplan.cross_far(far_seam(cplan))
        cross_ok, _ = cross_verdict(cplan)
    scale_carry = force_scale is not None
    out = provider.differentiable_accelerations(
        topology,
        positions,
        masses,
        params,
        plan=plan,
        force_scale=scale_carry,
        cross=cross,
    )
    acc, f_b = out if scale_carry else (out, None)
    acc = jnp.asarray(acc)
    if force_scale is not None:
        # as `fused._add_cross_near_term` takes the local f_b
        f_b = jnp.asarray(f_b, force_scale.dtype)
    if cplan is not None:
        tree = prepared.tree
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        x_sorted, m_sorted = engine._sorted_inputs(prepared, positions, masses)
        force = provider.resolve_params(params)
        eps2, g = _near_scalars(engine, force, acc.dtype)
        from jaccpot._env import env_flag

        term = near_seam(
            cplan,
            x_sorted,
            m_sorted,
            leaf_ranges=jnp.asarray(tree.node_ranges)[num_internal:],
            particle_indices=jnp.asarray(tree.particle_indices),
            num_internal=num_internal,
            leaf_width=max(
                int(jnp.asarray(prepared.nearfield_leaf_particle_indices).shape[1]),
                int(cplan.leaf_width),
            ),
            softening_sq=eps2,
            G=g,
            softening_kernel=getattr(engine, "softening_kernel", None),
            interpret=env_flag("JACCPOT_NEARFIELD_PALLAS_INTERPRET", False),
            with_force_scale=scale_carry,
        )
        if scale_carry:
            term, scale_term = term
            assert force_scale is not None and f_b is not None
            f_b = f_b + jnp.asarray(scale_term, force_scale.dtype)
        acc = acc + term
    live_rows = jnp.arange(int(positions.shape[0])) < live
    acc = jnp.where(live_rows[:, None], acc, jnp.zeros((), acc.dtype))
    # the force's verdict on the template's placeholder far list, as `step`'s
    bits, needs = _shard_verdict(
        replace(topology, prepared=replace(prepared, compact_far_pairs=None)),
        cross_hook,
        cross_near_sink,
    )
    lists_ok = local_symmetric & cross_ok
    bits = bits | jnp.where(lists_ok, jnp.uint32(0), jnp.uint32(OVERFLOW_GRAD_LISTS))
    ok = lists_ok & overflow_ok(bits)
    if cross_hook is not None:
        # one verdict for the mesh: a device's cotangents reach its peers' sources
        ok = jax.lax.pmin(ok.astype(jnp.int32), axis_name) > 0
    acc = _gradient_gate(acc, ok)
    if f_b is not None:
        assert force_scale is not None
        f_b = stop(jnp.where(live_rows, f_b, jnp.zeros((), f_b.dtype)))
        return acc, bits, needs, f_b
    return acc, bits, needs


# --------------------------------------------------------------------------
# The user-facing program (G4).
# --------------------------------------------------------------------------


def make_mesh_grad_fn(force: Any) -> Callable[..., tuple[Array, ...]]:
    """The mesh force as one differentiable function of mesh-sharded arrays.

    Wraps :meth:`~jaccpot.distributed.template_force.FusedMeshForce.differentiable_shard`
    in the force's ``shard_map`` (``check_vma=False``), so a scalar loss built on its
    output OUTSIDE the ``shard_map`` differentiates with plain ``jax.grad`` in the
    positions, the masses and -- passed as a
    :class:`~jaccpot.runtime.fused_topology.ForceParams` of arrays -- ``G`` and the
    softening squared (replicated: their cotangents are summed over the mesh by the
    ``shard_map`` transpose). Trace and call it as the force's own programs:
    under ``force.context()``, with ``force.before_trace()`` before each new trace,
    after ``force.prepare(...)`` and ``force.with_grad_plan(...)``.

    Parameters
    ----------
    force : Any
        A prepared :class:`~jaccpot.distributed.template_force.FusedMeshForce` with
        its gradient statics.

    Returns
    -------
    Callable[..., tuple[Array, ...]]
        ``f(positions, masses, counts, ids, params=None, force_scale=None) ->
        (acceleration, bits)`` (and ``f_b``, stopped, with ``force_scale``):
        ``(ndev * cap, 3)`` accelerations, zero on padding rows, NaN gradient when
        a device's lists are not ones the reverses read (``bits`` says why);
        ``bits`` ``(ndev,)``.
    """
    from jax.sharding import PartitionSpec as P

    spec = P(force.axis_name)

    def f(
        positions: Array,
        masses: Array,
        counts: Array,
        ids: Array,
        params: Any = None,
        force_scale: Optional[Array] = None,
    ) -> tuple[Array, ...]:
        extra = () if force_scale is None else (force_scale,)

        def body(x: Array, m: Array, n: Array, i: Array, p: Any, *fb: Array) -> Any:
            out = force.differentiable_shard(
                x, m, jnp.minimum(n[0], x.shape[0]), i, *fb, params=p
            )
            return (out[0], out[1].reshape(1), *out[3:])

        return jax.shard_map(
            body,
            mesh=force.mesh,
            in_specs=(spec, spec, spec, spec, P(), *((spec,) * len(extra))),
            out_specs=(spec, spec, *((spec,) * len(extra))),
            check_vma=False,
        )(positions, masses, counts, ids, params, *extra)

    return f
