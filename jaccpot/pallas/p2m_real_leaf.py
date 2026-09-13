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

__all__ = ["p2m_real_leaves_pallas", "pallas_p2m_real_leaf_supported"]


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
    cidx = lax.broadcasted_iota(jnp.int32, (coeff_pad,), 0)
    out = jnp.zeros((coeff_pad,), dtype)
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
            u = r_pow[n] * pnm * azim * inv_denom
            coef = jnp.sum(mass * u)
            out = jnp.where(cidx == sh_index(n, m), coef, out)
    out_ref[0, :] = out


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
