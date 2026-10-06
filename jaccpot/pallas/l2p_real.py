"""L2P (real basis): the far-field gradient at every particle, one particle per lane.

:func:`jaccpot.runtime.kernels._evaluate._evaluate_local_expansions_particle_major`
evaluated :func:`jaccpot.operators.real_p2m_l2p.evaluate_local_real_with_grad` per
particle under ``vmap``: spherical angles (square roots, divisions), the Legendre
recursion and reverse-mode autodiff over it, in XLA fusions over chunks of 2^21
particles that hand their intermediates through memory. Measured on an A100 (p6,
Plummer clipped, cell_min_level 8): ~4.4 ms per step at 8e6, ~52 ms at 1e8, for
~500 flops per particle.

Here a program owns ``P`` consecutive (tree-ordered) particles, each lane reads its
own leaf's expansion (lanes of one leaf hit the same rows) and the gradient comes
from the Cartesian recurrence of the complex inner solid harmonics
``Y_n^m = r^n P_n^m(cos t) e^{i m phi} / (n+m)!`` (no Condon-Shortley phase, the
basis of ``evaluate_local_real``: ``U_n^m = Re Y_n^|m|`` for ``m >= 0``, ``Im`` for
``m < 0``)::

    Y_m^m = (x + i y) / (2m) Y_{m-1}^{m-1}
    Y_n^m = ((2n - 1) z Y_{n-1}^m - r^2 Y_{n-2}^m) / ((n + m)(n - m))

and their derivatives, which are harmonics one degree lower::

    d_z Y_n^m           = Y_{n-1}^m
    (d_x - i d_y) Y_n^m = Y_{n-1}^{m-1}      (m = 0: -conj(Y_{n-1}^1))
    (d_x + i d_y) Y_n^m = -Y_{n-1}^{m+1}

so the gradient needs the harmonics to degree ``p - 1`` and no square root or
division by the radius (the axis and ``delta = 0`` are regular points). The same
gradient as the autodiff form to float rounding (1.9e-15 relative in float64,
1.2e-6 in float32). A leaf-major form (one program per block of leaves, their
particles a few lanes at a time) ran the tiles ~1/3 full on cell leaves and was no
faster in the step.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.operators._sh_indexing import sh_index
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = ["l2p_real_grad_cartesian", "l2p_real_particles_pallas"]


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def l2p_real_grad_cartesian(coeff: Any, x: Any, y: Any, z: Any, *, order: int) -> tuple:
    """``grad_delta sum_{n,m} F_n^m U_n^m(delta)`` by the Cartesian recurrence.

    Works on any broadcastable arrays (a kernel's tiles, or plain ``jnp`` arrays).

    Parameters
    ----------
    coeff : Any
        Indexable by packed index (:func:`~jaccpot.operators._sh_indexing.sh_index`):
        ``coeff[k]`` is the ``F`` of coefficient ``k`` (broadcastable to ``x``).
    x : Any
        ``delta_x = centre_x - position_x``.
    y : Any
        ``delta_y``.
    z : Any
        ``delta_z``.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    tuple
        ``(gx, gy, gz)``, the gradient with respect to ``delta``.
    """
    p = int(order)
    zero = x * 0
    gx, gy, gz = zero, zero, zero
    if p < 1:
        return gx, gy, gz
    r2 = x * x + y * y + z * z
    # Y[m] holds (re, im) of Y_{n-1}^m for the current n; prev of Y_{n-2}^m
    Y = {0: (zero + 1, zero)}
    prev: dict = {}
    for n in range(1, p + 1):
        d = n - 1  # the degree of the harmonics the gradient of degree n reads
        if d >= 1:
            new = {}
            for m in range(0, d):
                a1, b1 = Y[m]
                c = 1.0 / ((d + m) * (d - m))
                k = 2.0 * d - 1.0
                if m in prev:
                    a2, b2 = prev[m]
                    new[m] = ((k * z * a1 - r2 * a2) * c, (k * z * b1 - r2 * b2) * c)
                else:
                    new[m] = (k * z * a1 * c, k * z * b1 * c)
            a, b = Y[d - 1]
            new[d] = ((x * a - y * b) * (0.5 / d), (x * b + y * a) * (0.5 / d))
            prev, Y = Y, new
        for m in range(0, n + 1):
            fr = coeff[sh_index(n, m)]
            fi = coeff[sh_index(n, -m)] if m > 0 else None
            # d_z: Y_{n-1}^m
            if m <= d:
                zr, zi = Y[m]
                gz = gz + fr * zr
                if fi is not None:
                    gz = gz + fi * zi
            # A = Y_{n-1}^{m+1}, B = Y_{n-1}^{m-1} (m = 0: -conj(Y_{n-1}^1))
            ar, ai = Y[m + 1] if m + 1 <= d else (None, None)
            if m >= 1:
                br, bi = Y[m - 1]
            elif d >= 1:
                br, bi = -Y[1][0], Y[1][1]
            else:
                br, bi = None, None
            # d_x = (B - A) / 2,  d_y = i (A + B) / 2
            sr = (br if br is not None else 0) - (ar if ar is not None else 0)
            si = (bi if bi is not None else 0) - (ai if ai is not None else 0)
            tr = (ar if ar is not None else 0) + (br if br is not None else 0)
            ti = (ai if ai is not None else 0) + (bi if bi is not None else 0)
            gx = gx + fr * (0.5 * sr)
            gy = gy - fr * (0.5 * ti)
            if fi is not None:
                gx = gx + fi * (0.5 * si)
                gy = gy + fi * (0.5 * tr)
    return gx, gy, gz


def _l2p_particle_kernel(
    pos_ref: KernelRef,
    cent_ref: KernelRef,
    coef_ref: KernelRef,
    leaf_node_ref: KernelRef,
    leaf_end_ref: KernelRef,
    first_ref: KernelRef,
    live_ref: KernelRef,
    out_in: KernelRef,
    out_ref: KernelRef,
    *,
    order: int,
    block: int,
    rows: int,
    num_leaves: int,
) -> None:
    """``P`` consecutive particles, each in its own leaf's expansion (every lane busy).

    A particle's leaf is the first leaf that ends past it. The block's particles lie
    in at most ``P`` consecutive leaves from ``first_ref[block]`` on (every live leaf
    holds a particle), so each lane bisects that window -- no per-particle leaf
    array.

    Parameters
    ----------
    pos_ref : KernelRef
        Whole sorted positions ``[N, 3]``.
    cent_ref : KernelRef
        Whole expansion centres ``[nodes, 3]``.
    coef_ref : KernelRef
        Whole local coefficients ``[nodes, (p+1)^2]``.
    leaf_node_ref : KernelRef
        Whole ``[L]`` node id of each leaf.
    leaf_end_ref : KernelRef
        Whole ``[L]`` one past each leaf's last particle (non-decreasing).
    first_ref : KernelRef
        Whole ``[blocks]`` leaf of each block's first particle.
    live_ref : KernelRef
        ``[1]`` particles in the leaves (a prefix of the rows).
    out_in : KernelRef
        The zeroed ``[N, 3]`` gradient, aliased to ``out_ref`` (not read).
    out_ref : KernelRef
        **Output** the gradient rows.
    order : int
        Expansion order. Static.
    block : int
        ``P`` (a multiple of 32). Static.
    rows : int
        ``N``: lanes past it (and past the leaves' particles) store nothing. Static.
    num_leaves : int
        ``L``. Static.

    Returns
    -------
    None
        Writes the block's rows.
    """
    del out_in
    b = pl.program_id(0)
    lane = lax.broadcasted_iota(jnp.int32, (block,), 0)
    i = b * block + lane
    first = first_ref[b]
    # the leaf of each lane: the first leaf of the window [first, first + P) that
    # ends past it, by bisection (log2 P loads per lane)
    lo = jnp.full((block,), first, jnp.int32)
    hi = jnp.full((block,), first + block, jnp.int32)
    for _ in range(int(block).bit_length()):
        mid = (lo + hi) // 2
        end_mid = leaf_end_ref[jnp.minimum(mid, num_leaves - 1)]
        past = (end_mid <= i) & (mid < num_leaves)
        lo = jnp.where(past, mid + 1, lo)
        hi = jnp.where(past, hi, mid)
    leaf = lo
    live = (i < live_ref[0]) & (leaf < num_leaves)
    nd = jnp.where(live, leaf_node_ref[jnp.where(live, leaf, 0)], 0)
    ii = jnp.where(live, i, 0)
    x = cent_ref[nd, 0] - pos_ref[ii, 0]
    y = cent_ref[nd, 1] - pos_ref[ii, 1]
    z = cent_ref[nd, 2] - pos_ref[ii, 2]
    C = (int(order) + 1) ** 2
    coeff = [coef_ref[nd, k] for k in range(C)]
    gx, gy, gz = l2p_real_grad_cartesian(coeff, x, y, z, order=order)
    dest = jnp.where(live, i, rows + lane)
    plgpu.store(out_ref.at[dest, 0], gx.astype(out_ref.dtype), mask=live)
    plgpu.store(out_ref.at[dest, 1], gy.astype(out_ref.dtype), mask=live)
    plgpu.store(out_ref.at[dest, 2], gz.astype(out_ref.dtype), mask=live)


def l2p_real_particles_pallas(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    *,
    order: int,
    block: int = 128,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """``[N, 3]`` far-field gradient (with respect to ``delta``), one particle per lane.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]`` real local coefficients.
    centers : Array
        ``[nodes, 3]`` expansion centres.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    leaf_nodes : Array
        ``[L]`` node id of each leaf, in particle order: leaf ``i``'s particles
        directly follow leaf ``i - 1``'s, empty leaves only at the end.
    leaf_start : Array
        ``[L]`` first particle of each leaf.
    leaf_count : Array
        ``[L]`` particles of each leaf (0: empty).
    order : int
        Expansion order. Static.
    block : int
        Particles per program (a power of two, at least 32). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    Array
        ``[N, 3]`` gradient with respect to ``delta``, in the positions' dtype; zero
        for rows past the leaves' particles.

    Raises
    ------
    RuntimeError
        If Pallas is not available.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    n = int(positions_sorted.shape[0])
    L = int(leaf_nodes.shape[0])
    P = max(32, _next_pow2(block))
    blocks = -(-n // P)
    idx = jnp.asarray(leaf_nodes).dtype
    count = jnp.asarray(leaf_count, idx)
    end = jnp.asarray(leaf_start, idx) + count
    # the leaf of each block's first particle: leaves ending at or before it
    first = jnp.searchsorted(
        end, jnp.arange(blocks, dtype=idx) * P, side="right", method="scan_unrolled"
    ).astype(idx)
    live = jnp.reshape(jnp.sum(count).astype(idx), (1,))
    pos = jnp.asarray(positions_sorted)
    coef = jnp.asarray(coefficients)
    cent = jnp.asarray(centers, coef.dtype)
    out0 = jnp.zeros((n, 3), pos.dtype)
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    leaf_nodes = jnp.asarray(leaf_nodes, idx)
    return pl.pallas_call(
        functools.partial(
            _l2p_particle_kernel, order=int(order), block=P, rows=n, num_leaves=L
        ),
        grid=(blocks,),
        in_specs=[
            _full(pos),
            _full(cent),
            _full(coef),
            _full(leaf_nodes),
            _full(end),
            _full(first),
            _full(live),
            _full(out0),
        ],
        out_specs=_full(out0),
        out_shape=jax.ShapeDtypeStruct(out0.shape, out0.dtype),
        input_output_aliases={7: 0},
        interpret=bool(interpret),
        name=f"l2p_real_particle_p{int(order)}_b{P}",
        **backend_kwargs,
    )(pos, cent, coef, leaf_nodes, end, first, live, out0)
