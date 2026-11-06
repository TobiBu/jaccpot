"""Reverse of the fused multi-GPU lane's merged M2L launch (WP-G seam A).

Inside the mesh force's ``shard_map`` every device runs ONE M2L launch
(:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas`) over a
target-sorted CSR whose rows are its ``n_local`` tree nodes and whose sources are
local nodes (``< n_local``) or imported rows (``n_local + r``, ``r < recv_cap``):
the peers' multipoles and centres, received by one ragged exchange
(:func:`jaccpot.distributed.cross._symmetric_exchange`) whose payload row ``j`` is
the sender's node ``send_node[j]`` and whose cross pairs are merged into the local
rows (:func:`~jaccpot.pallas.m2l_real_csr.merge_into_csr`). The single-tree
custom_vjp (:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`)
refuses that call's reverse (``n_targets != rows``): the imported rows' owners live
on other devices.

**Why the reverse needs no transposed exchange.** The cross pair set is GLOBALLY
SYMMETRIC: the symmetric walk (``_symmetric_walk``) hands both ends of a far pair
the identical pair, so (my node ``a``, peer node ``b``) is in my row ``a`` (source:
my import row of ``b``) AND in the peer's row ``b`` (source: its import row of
``a``); the local pairs are symmetric within the device (the frozen list of a mutual
walk). The cotangent of a multipole ``M_a`` is ``sum_t J(c_t - c_a)^T lbar_t`` over
every target ``t`` of ``a``: by symmetry those are exactly the partners in row
``a`` -- local ones, whose ``lbar`` is here, and imported ones, whose ``lbar`` lives
on the peer. So each device sends the OUTPUT cotangent of the nodes it exported
FORWARD, along the forward's own layout (the same ``send_node`` rows, the same
``send_sizes``): import row ``r`` then holds the ``lbar`` of the very peer node whose
multipole sits in import row ``r``. With ``lbar_full = [lbar ; lbar_imported]`` the
existing symmetric sort-free reverse
(:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_reverse_pallas`,
``symmetric=True``) is exact on the local rows: its by-source pass (program ``a``,
partners as targets) gives ``mult_bar[a]`` and the source half of ``cent_bar[a]``,
its by-target pass the target half -- run here with the program grid over the
``n_local`` local rows only and the tables over all ``n_local + recv_cap`` rows. Each
device completes its own nodes' cotangents: no reverse scatter, no sort, no
atomics, deterministic. The imported rows get ZERO cotangent: their owners
accounted for them (the forward reads them ``stop_gradient``-ed).

The pair ``(a -> b)`` evaluated on the peer and the pair ``(b -> a)`` re-derived
here see the same ``delta`` only if the imported multipoles and centres are bitwise
the owners' arrays fed to their own launch (they are: the payload is
``upward.multipoles.{packed, centers}``, the downward's operands); a pair set that
is not symmetric -- an overflowed or mismatched exchange, a non-symmetric local list
-- gives a wrong gradient silently, so the caller gates on that (the plan's pair-hash
verdict).

The forward is the production launch on the same operands, bit for bit. Reverse
only (``custom_vjp``): no forward-mode rule.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.pallas._compat import pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import csr_by_target
from jaccpot.pallas.m2l_real_csr_lanes import (
    _lane_tables,
    _m2l_rev_lanes_kernel,
    m2l_real_csr_lanes_pallas,
)
from jaccpot.runtime._force_scale_levels import FarForceScaleInputs

__all__ = [
    "cross_far_m2l_cvjp",
    "cross_far_m2l_fb_cvjp",
    "cross_far_m2l_reverse_pallas",
    "imported_local_cotangents",
]


def _rows_error(multipoles: Array, n_local: int, recv_cap: int) -> Optional[str]:
    """Why a multipole array is not ``[local ; imported]``, or ``None``.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` the launch's multipoles.
    n_local : int
        Local (target) rows. Static.
    recv_cap : int
        Imported rows. Static.

    Returns
    -------
    Optional[str]
        The message when ``n != n_local + recv_cap`` or either is negative.
    """
    n = int(multipoles.shape[0])
    if int(n_local) < 0 or int(recv_cap) < 0 or n != int(n_local) + int(recv_cap):
        return (
            f"cross M2L seam: multipoles have {n} rows, expected n_local "
            f"({n_local}) + recv_cap ({recv_cap})"
        )
    return None


def cross_far_m2l_reverse_pallas(
    multipoles: Array,
    centers: Array,
    sources: Array,
    row_offsets: Array,
    loc_bar_full: Array,
    *,
    order: int,
    n_local: int,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
) -> tuple[Array, Array]:
    """The symmetric sort-free M2L reverse over the local rows of a merged CSR.

    :func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_reverse_pallas`
    with ``symmetric=True`` -- the same kernel
    (:func:`~jaccpot.pallas.m2l_real_csr_lanes._m2l_rev_lanes_kernel`), the same
    two passes over the forward's CSR (by source, then by target) -- with the
    program grid over the first ``n_local`` rows only, while the multipoles,
    centres and local cotangents it gathers per lane cover every row. Rows past
    ``n_local`` get no program and a zero cotangent.

    Exact on the local rows when every pair of a local row's list is also listed,
    reversed, in its partner's row -- locally, or on the partner's owner device --
    and ``loc_bar_full`` holds every row's local cotangent (the imported rows':
    their owners', :func:`imported_local_cotangents`). With ``n_local`` equal to
    the row count it IS the symmetric reverse, bit for bit.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles (the forward's, ``[local ; imported]``).
    centers : Array
        ``[n, 3]`` expansion centres, aligned.
    sources : Array
        ``[P]`` sources of the forward's target-sorted CSR (the merged list).
    row_offsets : Array
        Its row offsets, ``[>= n_local + 1]``.
    loc_bar_full : Array
        ``[n, C]`` cotangent of every row's local expansion.
    order : int
        Expansion order. Static.
    n_local : int
        Rows that own a program (the target rows). Static.
    k_lanes : int
        Pairs per iteration. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    tuple[Array, Array]
        ``(multipoles_bar [n, C], centers_bar [n, 3])``, zero past ``n_local``.

    Raises
    ------
    ValueError
        On a shape mismatch or ``n_local`` outside ``[0, n]``.
    """
    p = int(order)
    tables = _lane_tables(p)
    C = tables["C"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    n = int(mult.shape[0])
    nl = int(n_local)
    if not (0 <= nl <= n):
        raise ValueError(f"n_local must lie in [0, {n}], got {nl}")
    if int(mult.shape[1]) != C:
        raise ValueError(f"multipoles must have {C} coefficients for order {p}")
    cent = jnp.asarray(centers, dtype=dtype)
    if cent.shape != (n, 3):
        raise ValueError("centers must have shape (n, 3) aligned with multipoles")
    lbar = jnp.asarray(loc_bar_full, dtype)
    if lbar.shape != (n, C):
        raise ValueError(f"loc_bar_full must have shape ({n}, {C})")
    cent_p = jnp.pad(cent, ((0, 0), (0, 1)))
    src_sorted, off_t, cnt_t = csr_by_target(
        sources, row_offsets, total_nodes=nl, presorted=True
    )
    if nl == 0 or int(src_sorted.shape[0]) == 0:
        return jnp.zeros((n, C), dtype), jnp.zeros_like(jnp.asarray(centers))
    K = int(k_lanes)
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    def one_pass(*, by_target: bool) -> tuple[Array, Array]:
        kernel = functools.partial(
            _m2l_rev_lanes_kernel, p=p, k_lanes=K, tables=tables, by_target=by_target
        )
        operands = [mult, cent_p, src_sorted, off_t, cnt_t, lbar]
        return pl.pallas_call(
            kernel,
            grid=(nl,),
            in_specs=[bs_full(o) for o in operands],
            out_specs=[
                pl.BlockSpec((1, C), lambda s: (s, 0)),
                pl.BlockSpec((1, 4), lambda s: (s, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((nl, C), dtype),
                jax.ShapeDtypeStruct((nl, 4), dtype),
            ],
            interpret=bool(interpret),
            name=f"m2l_rev_real_csr_lanes_cross_{'tgt' if by_target else 'src'}"
            f"_p{p}_k{K}",
            **backend_kwargs,
        )(*operands)

    # row a lists the partners of a at both ends of a pair (locally, or on the
    # partner's owner): by source the multipole cotangents and the source half,
    # by target the target half
    mbar, cbar_src = one_pass(by_target=False)
    _, cbar_tgt = one_pass(by_target=True)
    cbar = (cbar_src[:, :3] + cbar_tgt[:, :3]).astype(jnp.asarray(centers).dtype)
    pad = n - nl
    mbar = jnp.concatenate([mbar, jnp.zeros((pad, C), dtype)])
    cbar = jnp.concatenate([cbar, jnp.zeros((pad, 3), cbar.dtype)])
    return mbar, cbar


def imported_local_cotangents(
    loc_bar: Array,
    send_node: Array,
    alive: Array,
    send_sizes: Array,
    *,
    recv_cap: int,
    axis_name: str,
) -> Array:
    """The peers' local cotangents of the nodes this device imported, row for row.

    The forward's far payload, with each exported node's local cotangent in place
    of its multipole: row ``j`` is ``loc_bar[send_node[j]]`` (zero where not
    ``alive``), sent with the forward's ``send_sizes`` by the same exchange
    (``ragged_all_to_all_exchange``, the same direction -- never a transpose). So
    row ``r`` of the result is the local cotangent, on its owner, of the node whose
    multipole this device holds in import row ``r``. Call inside ``shard_map``.

    Parameters
    ----------
    loc_bar : Array
        ``[n_local, C]`` this device's local cotangents.
    send_node : Array
        ``[S]`` the tree node behind each export row (``nodes[max(exp_f.rows,
        0)]`` in ``_symmetric_exchange``).
    alive : Array
        ``[S]`` (or ``[S, 1]``) live export rows (``exp_f.rows >= 0``).
    send_sizes : Array
        ``[ndev]`` export rows per peer (``exp_f.sizes``).
    recv_cap : int
        The receive buffer's rows (``cap.recv_node_cap``). Static.
    axis_name : str
        Mesh axis. Static.

    Returns
    -------
    Array
        ``[recv_cap, C]``, ``loc_bar``'s dtype; zero past the received rows.
    """
    from yggdrax.distributed.comm import ragged_all_to_all_exchange

    lb = jnp.asarray(loc_bar)
    node = jnp.asarray(send_node).astype(jnp.int32).reshape(-1)
    live = jnp.asarray(alive).reshape(-1)
    lb_send = jnp.where(live[:, None], lb[jnp.maximum(node, 0)], 0.0).astype(lb.dtype)
    got, _, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_far_cotangent"
    )(lb_send, send_sizes, output_capacity=int(recv_cap), axis_name=axis_name)
    return got


def _cross_backward(
    residual: tuple,
    loc_bar: Array,
    *,
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_local: int,
    recv_cap: int,
    axis_name: str,
) -> tuple[Array, Array]:
    """The seam's reverse, shared by both variants.

    Parameters
    ----------
    residual : tuple
        ``(multipoles, centers, sources, row_offsets, send_node, alive,
        send_sizes)`` from the forward.
    loc_bar : Array
        ``[n_local, C]`` cotangent of the local increments.
    order : int
        Expansion order. Static.
    k_lanes : int
        Pairs per iteration. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    n_local : int
        Local rows. Static.
    recv_cap : int
        Imported rows. Static.
    axis_name : str
        Mesh axis. Static.

    Returns
    -------
    tuple[Array, Array]
        ``(multipoles_bar, centers_bar)`` over ``[local ; imported]``, zero on the
        imported rows.
    """
    multipoles, centers, sources, row_offsets, send_node, alive, send_sizes = residual
    lb = jnp.asarray(loc_bar, jnp.asarray(multipoles).dtype)
    lb_imp = imported_local_cotangents(
        lb, send_node, alive, send_sizes, recv_cap=recv_cap, axis_name=axis_name
    )
    return cross_far_m2l_reverse_pallas(
        multipoles,
        centers,
        sources,
        row_offsets,
        jnp.concatenate([lb, lb_imp.astype(lb.dtype)]),
        order=order,
        n_local=n_local,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def cross_far_m2l_cvjp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    row_offsets: Array,
    active_pair_count: Optional[Array],
    send_node: Array,
    alive: Array,
    send_sizes: Array,
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_local: int,
    recv_cap: int,
    axis_name: str,
) -> Array:
    """The device's merged M2L launch, differentiable across devices (seam A).

    Forward: :func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas`
    on ``(multipoles, centers, sources, row_offsets)`` with ``n_targets=n_local``
    and ``presorted=True`` -- the production call (``_downward_prep.
    _solidfmm_downward_accumulate_from_multipoles``' CSR lane after
    ``merge_into_csr``), bit for bit. Reverse: the local cotangents of the exported
    nodes go forward along the far exchange's layout
    (:func:`imported_local_cotangents`), then the symmetric sort-free reverse runs
    over the local rows (:func:`cross_far_m2l_reverse_pallas`); see the module
    docstring for why that is exact on a globally symmetric pair set. Call inside
    ``shard_map`` (the reverse holds one ragged exchange).

    Parameters
    ----------
    multipoles : Array
        ``[n_local + recv_cap, C]`` real multipoles, ``[local ; imported]`` (the
        imported rows ``stop_gradient``-ed by the caller). Differentiable: the
        imported rows' cotangent is zero.
    centers : Array
        ``[n_local + recv_cap, 3]`` expansion centres, aligned. Differentiable, the
        same way.
    sources : Array
        ``[P]`` sources of the merged target-sorted CSR (``merge_into_csr``'s
        first output). Topology: no cotangent.
    row_offsets : Array
        Its row offsets ``[n_local + 1]`` (``merge_into_csr``'s second). Topology.
    active_pair_count : Optional[Array]
        The merged live count (``merge_into_csr``'s third), or ``None``; unused by
        the CSR read (``presorted``), carried for the call's shape. Topology.
    send_node : Array
        ``[send_cap]`` int, the tree node behind each far export row:
        ``nodes[max(exp_f.rows, 0)]`` in ``cross._symmetric_exchange``. Topology.
    alive : Array
        ``[send_cap]`` (or ``[send_cap, 1]``) bool, ``exp_f.rows >= 0``. Topology.
    send_sizes : Array
        ``[ndev]`` int, ``exp_f.sizes`` (the forward far exchange's send sizes).
        Topology.
    order : int
        Expansion order. ``nondiff_argnums``.
    k_lanes : int
        Pairs per iteration (``JACCPOT_M2L_CSR_LANES``). ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode (``JACCPOT_M2L_CSR_INTERPRET``). ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering (``"triton"``). ``nondiff_argnums``.
    num_warps : int
        Warps per program (``JACCPOT_M2L_CSR_WARPS``). ``nondiff_argnums``.
    n_local : int
        Local nodes = target rows (the forward's ``n_targets``). ``nondiff_argnums``.
    recv_cap : int
        Imported rows (``cap.recv_node_cap``, the far exchange's
        ``output_capacity``). ``nondiff_argnums``.
    axis_name : str
        Mesh axis of the far exchange. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[n_local, C]`` local increments.

    Raises
    ------
    ValueError
        If ``multipoles`` does not have ``n_local + recv_cap`` rows.
    """
    error = _rows_error(multipoles, n_local, recv_cap)
    if error is not None:
        raise ValueError(error)
    return m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        row_offsets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=int(n_local),
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=True,
    )


def _cross_cvjp_fwd(
    multipoles,
    centers,
    sources,
    row_offsets,
    active_pair_count,
    send_node,
    alive,
    send_sizes,
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_local,
    recv_cap,
    axis_name,
):
    out = cross_far_m2l_cvjp(
        multipoles,
        centers,
        sources,
        row_offsets,
        active_pair_count,
        send_node,
        alive,
        send_sizes,
        order,
        k_lanes,
        interpret,
        backend,
        num_warps,
        n_local,
        recv_cap,
        axis_name,
    )
    return out, (
        multipoles,
        centers,
        sources,
        row_offsets,
        send_node,
        alive,
        send_sizes,
    )


def _cross_cvjp_bwd(
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_local,
    recv_cap,
    axis_name,
    residual,
    loc_bar,
):
    mult_bar, cent_bar = _cross_backward(
        residual,
        loc_bar,
        order=order,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        n_local=n_local,
        recv_cap=recv_cap,
        axis_name=axis_name,
    )
    return (mult_bar, cent_bar, None, None, None, None, None, None)


cross_far_m2l_cvjp.defvjp(_cross_cvjp_fwd, _cross_cvjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(12, 13, 14, 15, 16, 17, 18, 19, 20))
def cross_far_m2l_fb_cvjp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    row_offsets: Array,
    active_pair_count: Optional[Array],
    send_node: Array,
    alive: Array,
    send_sizes: Array,
    node_mass: Array,
    node_radii: Array,
    softening_sq: Array,
    node_centers: Optional[Array],
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_local: int,
    recv_cap: int,
    axis_name: str,
    gravitational_constant: float,
) -> tuple[Array, Array]:
    """:func:`cross_far_m2l_cvjp` for ``mac_type='dehnen_error'``'s launch.

    Forward: the force-scale launch
    (:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_fb_pallas_cvjp`'s
    forward with ``n_targets=n_local``, ``presorted=True``), bit for bit: the local
    increments and eq (16b)'s far term per local row. Reverse: the increments' as
    :func:`cross_far_m2l_cvjp`; the far term is topology (the next walk's
    thresholds, frozen in the gradient), its cotangent dropped and its inputs given
    none -- as the single-tree variant does. The reverse is exact only when the
    merged list is symmetric (cross pairs are by the symmetric walk; the LOCAL
    criterion list must be checked by the caller, which the single-tree variant
    does not assume).

    Parameters
    ----------
    multipoles : Array
        As :func:`cross_far_m2l_cvjp`. Differentiable.
    centers : Array
        As :func:`cross_far_m2l_cvjp`. Differentiable.
    sources : Array
        As :func:`cross_far_m2l_cvjp`. Topology.
    row_offsets : Array
        As :func:`cross_far_m2l_cvjp`. Topology.
    active_pair_count : Optional[Array]
        As :func:`cross_far_m2l_cvjp`. Topology.
    send_node : Array
        As :func:`cross_far_m2l_cvjp`. Topology.
    alive : Array
        As :func:`cross_far_m2l_cvjp`. Topology.
    send_sizes : Array
        As :func:`cross_far_m2l_cvjp`. Topology.
    node_mass : Array
        :attr:`FarForceScaleInputs.node_mass` over ``[local ; imported]`` (the
        imported rows' masses travel in the far payload). Not differentiated.
    node_radii : Array
        :attr:`FarForceScaleInputs.node_radii` (``[local ; imported]``; only the
        local targets' are read). Not differentiated.
    softening_sq : Array
        :attr:`FarForceScaleInputs.softening_sq`. Not differentiated.
    node_centers : Optional[Array]
        :attr:`FarForceScaleInputs.node_centers`. Not differentiated.
    order : int
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    k_lanes : int
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    interpret : bool
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    backend : str
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    num_warps : int
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    n_local : int
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    recv_cap : int
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    axis_name : str
        As :func:`cross_far_m2l_cvjp`. ``nondiff_argnums``.
    gravitational_constant : float
        :attr:`FarForceScaleInputs.gravitational_constant`. ``nondiff_argnums``.

    Returns
    -------
    tuple[Array, Array]
        ``([n_local, C]`` local increments, ``[n_local]`` far term ``)``.

    Raises
    ------
    ValueError
        If ``multipoles`` does not have ``n_local + recv_cap`` rows.
    """
    error = _rows_error(multipoles, n_local, recv_cap)
    if error is not None:
        raise ValueError(error)
    out = m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        row_offsets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=int(n_local),
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=True,
        force_scale=FarForceScaleInputs(
            node_mass=node_mass,
            node_radii=node_radii,
            softening_sq=softening_sq,
            gravitational_constant=gravitational_constant,
            node_centers=node_centers,
        ),
    )
    return out[0], out[1]


def _cross_fb_cvjp_fwd(
    multipoles,
    centers,
    sources,
    row_offsets,
    active_pair_count,
    send_node,
    alive,
    send_sizes,
    node_mass,
    node_radii,
    softening_sq,
    node_centers,
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_local,
    recv_cap,
    axis_name,
    gravitational_constant,
):
    out = cross_far_m2l_fb_cvjp(
        multipoles,
        centers,
        sources,
        row_offsets,
        active_pair_count,
        send_node,
        alive,
        send_sizes,
        node_mass,
        node_radii,
        softening_sq,
        node_centers,
        order,
        k_lanes,
        interpret,
        backend,
        num_warps,
        n_local,
        recv_cap,
        axis_name,
        gravitational_constant,
    )
    return out, (
        multipoles,
        centers,
        sources,
        row_offsets,
        send_node,
        alive,
        send_sizes,
    )


def _cross_fb_cvjp_bwd(
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_local,
    recv_cap,
    axis_name,
    gravitational_constant,
    residual,
    cotangents,
):
    loc_bar, _ = cotangents
    mult_bar, cent_bar = _cross_backward(
        residual,
        loc_bar,
        order=order,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        n_local=n_local,
        recv_cap=recv_cap,
        axis_name=axis_name,
    )
    return (mult_bar, cent_bar) + (None,) * 10


cross_far_m2l_fb_cvjp.defvjp(_cross_fb_cvjp_fwd, _cross_fb_cvjp_bwd)
