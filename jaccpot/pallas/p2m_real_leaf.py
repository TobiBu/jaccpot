"""Leaf P2M in the Dehnen real basis as ONE Pallas launch: one program per leaf (plan sub-10ms, Phase 3).

``_p2m_leaves_real`` evaluates ``p2m_real_direct`` under two ``vmap``s in
batches of 4096 leaves: XLA turns the unrolled per-coefficient recurrence into
long chains of small fusions over ``(4096, 64)`` tiles and transposes the
``(leaves, particles, coeffs)`` tensor to reduce it -- 9.4 ms per step at
N = 2x10^5 on 16k cell leaves (4 batches x ~2.3 ms), as much as the whole M2M.

Here one program owns one leaf: it loads its (up to) ``W`` particles as lane
vectors, evaluates the real regular solid harmonics ``U_n^m`` with the SAME
recurrences as :func:`jaccpot.operators.real_p2m_l2p.p2m_real_direct`
(Chebyshev ``cos(m phi)``/``sin(m phi)``, Legendre ``P_n^m`` without the
Condon-Shortley phase, ``1/(n+|m|)!`` normalisation, the same floored radii),
and reduces ``mass * U_n^m`` over the lanes -- 36 lane reductions per leaf at
``p = 5``. Static loops, static factorials, no tables. Agrees with the
reference to float32 summation order (the lane sum is a tree reduction).
"""

from __future__ import annotations

import functools
import math
from typing import Iterator

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.operators._sh_indexing import sh_index, sh_size
from jaccpot.operators.dtypes import squared_radius_floor
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None

__all__ = [
    "p2m_real_leaves_pallas",
    "p2m_real_leaves_pallas_cvjp",
    "p2m_real_leaves_reverse_pallas",
    "pallas_p2m_real_leaf_supported",
]


def pallas_p2m_real_leaf_supported() -> bool:
    """True where the Triton lowering runs (sm_80+, as the other real kernels).

    Returns
    -------
    bool
        Whether the kernel can run natively here.
    """
    return pallas_m2l_real_csr_supported()


#: Measured on an idle A100 (leaf 64 lanes, 16k cell leaves): the register
#: allocation is sensitive to the warp count and the wrong one spills.
_DEFAULT_WARPS = {4: 2, 5: 4, 6: 2}


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _p2m_leaf_kernel(
    pos_ref: KernelRef,
    mass_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    cent_ref: KernelRef,
    out_ref: KernelRef,
    *,
    order: int,
    width: int,
    coeff_pad: int,
    floor: float,
) -> None:
    """One leaf: ``M_n^m = sum_i m_i U_n^m(x_i - c)`` over its particle lanes.

    Parameters
    ----------
    pos_ref : KernelRef
        Whole sorted position table padded by ``width`` rows, ``[n + W, 3]``.
    mass_ref : KernelRef
        Whole sorted mass table, padded likewise, ``[n + W]``.
    start_ref : KernelRef
        First particle of each leaf ``[L]`` (``n`` for empty leaves).
    count_ref : KernelRef
        Particle count of each leaf ``[L]``.
    cent_ref : KernelRef
        Expansion centre of each leaf ``[L, 3]``.
    out_ref : KernelRef
        **Output** ``[1, coeff_pad]``: packed coefficients, zero past ``C``.
    order : int
        Expansion order ``p``. Static.
    width : int
        Leaf capacity ``W`` (lanes). Static.
    coeff_pad : int
        Output row width (power of two >= ``(p+1)^2``). Static.
    floor : float
        The dtype's squared-radius floor (``squared_radius_floor``). Static.

    Returns
    -------
    None
        Writes the leaf's row.
    """
    p = int(order)
    leaf = pl.program_id(0)
    start = start_ref[leaf]
    count = count_ref[leaf]
    lanes = pl.ds(start, width)
    dtype = out_ref.dtype
    x = pos_ref[lanes, 0] - cent_ref[leaf, 0]
    y = pos_ref[lanes, 1] - cent_ref[leaf, 1]
    z = pos_ref[lanes, 2] - cent_ref[leaf, 2]
    lane = lax.broadcasted_iota(jnp.int32, (width,), 0)
    valid = lane < count
    mass = jnp.where(valid, mass_ref[lanes], jnp.asarray(0.0, dtype))
    cidx = lax.broadcasted_iota(jnp.int32, (coeff_pad,), 0)
    out = jnp.zeros((coeff_pad,), dtype)
    # a generator, so each U_n^m is reduced as soon as it is formed (the trace
    # is the one the forward always had; the reverse consumes the same stream)
    for idx_c, u in _regular_harmonics_lanes(x, y, z, order=p, floor=floor):
        coef = jnp.sum(mass * u)
        out = jnp.where(cidx == idx_c, coef, out)
    out_ref[0, :] = out


def _regular_harmonics_lanes(
    x: Array, y: Array, z: Array, *, order: int, floor: float
) -> Iterator[tuple[int, Array]]:
    """Yield ``(packed index, U_n^m)`` for every ``(n, m)`` on lane vectors, in packed order.

    The recurrences of :func:`jaccpot.operators.real_p2m_l2p.p2m_real_direct`
    (Chebyshev ``cos(m phi)`` / ``sin(m phi)``, Legendre ``P_n^m`` without the
    Condon-Shortley phase, ``1/(n+|m|)!`` normalisation, the same floored radii).

    Parameters
    ----------
    x : Array
        Lane vector of ``x - c_x``.
    y : Array
        Lane vector of ``y - c_y``.
    z : Array
        Lane vector of ``z - c_z``.
    order : int
        Expansion order ``p``. Static.
    floor : float
        The dtype's squared-radius floor. Static.

    Yields
    ------
    tuple[int, Array]
        ``(sh_index(n, m), U_n^m)`` with ``U_n^m`` a lane vector.
    """
    p = int(order)
    dtype = x.dtype
    fl = jnp.asarray(floor, dtype)
    r2 = jnp.maximum(x * x + y * y + z * z, fl)
    r = jnp.sqrt(r2)
    rho = jnp.sqrt(jnp.maximum(x * x + y * y, fl))
    cos_t = z / r
    sin_t = rho / r
    cos_p = x / rho
    sin_p = y / rho
    one = jnp.ones_like(x)
    cosm = [one, cos_p]
    sinm = [jnp.zeros_like(x), sin_p]
    for m in range(2, p + 1):
        cosm.append(2.0 * cos_p * cosm[m - 1] - cosm[m - 2])
        sinm.append(2.0 * cos_p * sinm[m - 1] - sinm[m - 2])
    r_pow = [one]
    for n in range(1, p + 1):
        r_pow.append(r_pow[-1] * r)
    sin_pow = [one]
    for m in range(1, p + 1):
        sin_pow.append(sin_pow[-1] * sin_t)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            ma = abs(m)
            if ma == 0:
                pmm = one
            else:
                dfact = math.factorial(2 * ma) / ((2.0**ma) * math.factorial(ma))
                pmm = dfact * sin_pow[ma]
            if ma == n:
                pnm = pmm
            else:
                pnm2 = pmm
                pnm1 = (2.0 * ma + 1.0) * cos_t * pmm
                for k in range(ma + 2, n + 1):
                    pk = ((2.0 * k - 1.0) * cos_t * pnm1 - (k + ma - 1.0) * pnm2) / (
                        k - ma
                    )
                    pnm2, pnm1 = pnm1, pk
                pnm = pnm1
            inv_denom = 1.0 / math.factorial(n + ma)
            azim = cosm[ma] if m >= 0 else sinm[ma]
            yield sh_index(n, m), r_pow[n] * pnm * azim * inv_denom


def p2m_real_leaves_pallas(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    *,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int | None = None,
) -> Array:
    """Packed leaf multipoles for every leaf in one Pallas launch.

    Parameters
    ----------
    positions_sorted : Array
        ``[n, 3]`` positions in tree order.
    masses_sorted : Array
        ``[n]`` masses in tree order.
    leaf_centers : Array
        ``[L, 3]`` expansion centres of the leaves (``centers[num_internal:]``).
    leaf_ranges : Array
        ``[L, 2]`` inclusive particle ranges of the leaves (``node_ranges[num_internal:]``);
        an empty leaf has ``end < start``.
    order : int
        Expansion order. Static.
    num_internal : int
        Internal node count (rows left zero). Static.
    total_nodes : int
        Node count of the returned table. Static.
    leaf_width : int
        Leaf capacity (lanes per program). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int | None
        Warps per program; ``None`` picks the measured best per order
        (A100, 64 lanes, N=2e5 cell tree: p4 2 -> 0.22 ms, p5 4 -> 0.59 ms,
        p6 2 -> 0.46 ms; the wrong count spills and costs 4-5x).

    Returns
    -------
    Array
        ``[total_nodes, (p+1)^2]`` packed multipoles, internal rows zero.
    """
    p = int(order)
    C = sh_size(p)
    cp = _next_pow2(C)
    if num_warps is None:
        num_warps = _DEFAULT_WARPS.get(p, 2)
    dtype = jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    n = int(positions_sorted.shape[0])
    L = int(leaf_ranges.shape[0])
    w = int(leaf_width)
    pos = jnp.pad(jnp.asarray(positions_sorted, dtype), ((0, w), (0, 0)))
    mass = jnp.pad(jnp.asarray(masses_sorted, dtype), (0, w))
    ranges = jnp.asarray(leaf_ranges)
    counts = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0).astype(jnp.int32)
    starts = jnp.where(counts > 0, ranges[:, 0], n).astype(jnp.int32)
    cent = jnp.asarray(leaf_centers, dtype)
    kernel = functools.partial(
        _p2m_leaf_kernel,
        order=p,
        width=w,
        coeff_pad=cp,
        floor=float(squared_radius_floor(dtype)),
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr):
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    rows = pl.pallas_call(
        kernel,
        grid=(L,),
        in_specs=[_full(pos), _full(mass), _full(starts), _full(counts), _full(cent)],
        out_specs=pl.BlockSpec((1, cp), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((L, cp), dtype),
        interpret=bool(interpret),
        name=f"p2m_real_leaf_p{p}_w{w}",
        **backend_kwargs,
    )(pos, mass, starts, counts, cent)
    leaf_rows = rows[:, :C]
    return jnp.concatenate(
        [jnp.zeros((int(num_internal), C), dtype), leaf_rows], axis=0
    )[: int(total_nodes)]


# ------------------------------------------------------------------ reverse
# Adjoint of the leaf P2M (plan fast-gradients, Phase 2): ``M_n^m = sum_i m_i
# U_n^m(x_i - c)`` is linear in the masses and smooth in the positions, so with
# the leaf cotangent ``gbar`` the per-particle cotangents are
#
#     mass_bar_i = sum_nm gbar_nm U_nm(x_i - c)          (an L2P-shaped evaluation)
#     pos_bar_i  = m_i  grad_x [ sum_nm gbar_nm U_nm ]   (its gradient, per lane)
#     c_bar      = - sum_i pos_bar_i
#
# One program per leaf as the forward; the position gradient is ``jax.vjp`` of the
# lane-wise contraction with the mass lanes as cotangent (lanes are independent,
# so the lane-wise vjp IS the per-particle gradient), traced inside the kernel.
# The block output ``(L, W, 4)`` is scattered onto the particles in XLA (every
# particle belongs to exactly one leaf slot, so the scatter has unique indices).


def _p2m_rev_leaf_kernel(
    pos_ref: KernelRef,
    mass_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    cent_ref: KernelRef,
    gbar_ref: KernelRef,
    out_ref: KernelRef,
    *,
    order: int,
    width: int,
    floor: float,
) -> None:
    """One leaf: per-particle position and mass cotangents from the leaf's multipole cotangent.

    Parameters
    ----------
    pos_ref : KernelRef
        Sorted positions padded by ``width`` rows ``[n + W, 3]``.
    mass_ref : KernelRef
        Sorted masses, padded likewise ``[n + W]``.
    start_ref : KernelRef
        First particle per leaf ``[L]``.
    count_ref : KernelRef
        Particle count per leaf ``[L]``.
    cent_ref : KernelRef
        Leaf expansion centres ``[L, 3]``.
    gbar_ref : KernelRef
        Leaf multipole cotangents ``[L, C]``.
    out_ref : KernelRef
        **Output** ``[1, W, 4]``: lanes 0:3 the position cotangent, lane 3 the
        mass cotangent, zero on slots past the count.
    order : int
        Expansion order. Static.
    width : int
        Leaf capacity ``W``. Static.
    floor : float
        Squared-radius floor. Static.

    Returns
    -------
    None
        Writes the leaf's block.
    """
    p = int(order)
    leaf = pl.program_id(0)
    start = start_ref[leaf]
    count = count_ref[leaf]
    lanes = pl.ds(start, width)
    dtype = out_ref.dtype
    x = pos_ref[lanes, 0] - cent_ref[leaf, 0]
    y = pos_ref[lanes, 1] - cent_ref[leaf, 1]
    z = pos_ref[lanes, 2] - cent_ref[leaf, 2]
    lane = lax.broadcasted_iota(jnp.int32, (width,), 0)
    valid = lane < count
    mass = jnp.where(valid, mass_ref[lanes], jnp.asarray(0.0, dtype))
    g = [gbar_ref[leaf, sh_index(n, m)] for n in range(p + 1) for m in range(-n, n + 1)]

    def contraction(xx: Array, yy: Array, zz: Array) -> Array:
        acc = None
        for k, (_, u) in enumerate(
            _regular_harmonics_lanes(xx, yy, zz, order=p, floor=floor)
        ):
            term = g[k] * u
            acc = term if acc is None else acc + term
        return acc

    u_val, vjp = jax.vjp(contraction, x, y, z)
    xb, yb, zb = vjp(mass)
    zero = jnp.zeros_like(x)
    out_ref[0, :, 0] = jnp.where(valid, xb, zero)
    out_ref[0, :, 1] = jnp.where(valid, yb, zero)
    out_ref[0, :, 2] = jnp.where(valid, zb, zero)
    out_ref[0, :, 3] = jnp.where(valid, u_val, zero)


def p2m_real_leaves_reverse_pallas(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    packed_bar: Array,
    *,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int | None = None,
) -> tuple[Array, Array, Array]:
    """Adjoint of :func:`p2m_real_leaves_pallas`: one Pallas program per leaf.

    Parameters
    ----------
    positions_sorted : Array
        ``[n, 3]`` positions in tree order (the forward's).
    masses_sorted : Array
        ``[n]`` masses in tree order.
    leaf_centers : Array
        ``[L, 3]`` leaf expansion centres.
    leaf_ranges : Array
        ``[L, 2]`` inclusive particle ranges.
    packed_bar : Array
        ``[total_nodes, C]`` cotangent of the forward's output table; only the
        leaf rows matter (the internal rows of the output are constant zeros).
    order : int
        Expansion order. Static.
    num_internal : int
        Internal node count. Static.
    total_nodes : int
        Node count. Static.
    leaf_width : int
        Leaf capacity. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int | None
        Warps per program (``None`` = the forward's per-order default).

    Returns
    -------
    tuple[Array, Array, Array]
        ``(positions_bar [n, 3], masses_bar [n], leaf_centers_bar [L, 3])``.
    """
    p = int(order)
    C = sh_size(p)
    if num_warps is None:
        num_warps = _DEFAULT_WARPS.get(p, 2)
    dtype = jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    n = int(positions_sorted.shape[0])
    L = int(leaf_ranges.shape[0])
    w = int(leaf_width)
    pos = jnp.pad(jnp.asarray(positions_sorted, dtype), ((0, w), (0, 0)))
    mass = jnp.pad(jnp.asarray(masses_sorted, dtype), (0, w))
    ranges = jnp.asarray(leaf_ranges)
    counts = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0).astype(jnp.int32)
    starts = jnp.where(counts > 0, ranges[:, 0], n).astype(jnp.int32)
    cent = jnp.asarray(leaf_centers, dtype)
    gbar = jnp.asarray(packed_bar, dtype)[int(num_internal) : int(num_internal) + L, :C]
    if (
        int(gbar.shape[0]) < L
    ):  # total_nodes < num_internal + L: rows past the table are zero
        gbar = jnp.pad(gbar, ((0, L - int(gbar.shape[0])), (0, 0)))
    kernel = functools.partial(
        _p2m_rev_leaf_kernel, order=p, width=w, floor=float(squared_radius_floor(dtype))
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr):
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    out = pl.pallas_call(
        kernel,
        grid=(L,),
        in_specs=[
            _full(pos),
            _full(mass),
            _full(starts),
            _full(counts),
            _full(cent),
            _full(gbar),
        ],
        out_specs=pl.BlockSpec((1, w, 4), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((L, w, 4), dtype),
        interpret=bool(interpret),
        name=f"p2m_rev_real_leaf_p{p}_w{w}",
        **backend_kwargs,
    )(pos, mass, starts, counts, cent, gbar)
    lane = jnp.arange(w, dtype=jnp.int32)
    slot_valid = lane[None, :] < counts[:, None]
    idx = jnp.where(slot_valid, starts[:, None] + lane[None, :], n)  # dead row n
    buf = jnp.zeros((n + 1, 4), dtype).at[idx.reshape(-1)].add(out.reshape(-1, 4))
    positions_bar = buf[:n, :3].astype(positions_sorted.dtype)
    masses_bar = buf[:n, 3].astype(masses_sorted.dtype)
    centers_bar = (-jnp.sum(out[:, :, :3], axis=1)).astype(leaf_centers.dtype)
    return positions_bar, masses_bar, centers_bar


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def p2m_real_leaves_pallas_cvjp(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool,
    backend: str,
    num_warps: int | None,
) -> Array:
    """Differentiable :func:`p2m_real_leaves_pallas` (forward byte-identical).

    Parameters
    ----------
    positions_sorted : Array
        ``[n, 3]`` positions in tree order. Differentiable.
    masses_sorted : Array
        ``[n]`` masses in tree order. Differentiable.
    leaf_centers : Array
        ``[L, 3]`` leaf expansion centres. Differentiable.
    leaf_ranges : Array
        ``[L, 2]`` inclusive particle ranges (topology).
    order : int
        Expansion order. ``nondiff_argnums``.
    num_internal : int
        Internal node count. ``nondiff_argnums``.
    total_nodes : int
        Node count. ``nondiff_argnums``.
    leaf_width : int
        Leaf capacity. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int | None
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[total_nodes, C]`` packed multipoles, internal rows zero.
    """
    return p2m_real_leaves_pallas(
        positions_sorted,
        masses_sorted,
        leaf_centers,
        leaf_ranges,
        order=order,
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_width=leaf_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def _p2m_cvjp_fwd(
    positions_sorted,
    masses_sorted,
    leaf_centers,
    leaf_ranges,
    order,
    num_internal,
    total_nodes,
    leaf_width,
    interpret,
    backend,
    num_warps,
):
    out = p2m_real_leaves_pallas(
        positions_sorted,
        masses_sorted,
        leaf_centers,
        leaf_ranges,
        order=order,
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_width=leaf_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, (positions_sorted, masses_sorted, leaf_centers, leaf_ranges)


def _p2m_cvjp_bwd(
    order,
    num_internal,
    total_nodes,
    leaf_width,
    interpret,
    backend,
    num_warps,
    residual,
    packed_bar,
):
    positions_sorted, masses_sorted, leaf_centers, leaf_ranges = residual
    pos_bar, mass_bar, cent_bar = p2m_real_leaves_reverse_pallas(
        positions_sorted,
        masses_sorted,
        leaf_centers,
        leaf_ranges,
        packed_bar,
        order=order,
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_width=leaf_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return (pos_bar, mass_bar, cent_bar, None)


p2m_real_leaves_pallas_cvjp.defvjp(_p2m_cvjp_fwd, _p2m_cvjp_bwd)
