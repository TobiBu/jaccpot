"""L2P (real basis): the far-field gradient at every particle, one particle per lane.

:func:`jaccpot.runtime.kernels._evaluate._evaluate_local_expansions_particle_major`
evaluated :func:`jaccpot.operators.real_p2m_l2p.evaluate_local_real_with_grad` per
particle under ``vmap``: spherical angles (square roots, divisions), the Legendre
recursion and reverse-mode autodiff over it, in XLA fusions over chunks of 2^21
particles that hand their intermediates through memory. Measured on an A100 (p6,
Plummer clipped, cell_min_level 8): ~4.4 ms per step at 8e6, ~52 ms at 1e8, for
~500 flops per particle.

Here a program owns ``P`` consecutive (tree-ordered) particles, each lane reads its
own leaf's expansion (lanes of one leaf hit the same rows), and the expansion is
evaluated in the SAME basis the M2L / L2L produce its coefficients in: the real
inner solid harmonics ``U_n^m`` of :mod:`jaccpot.operators.real_p2m_l2p` (Dehnen
normalisation, no sqrt 2), contracted with the local coefficients exactly as
``evaluate_local_real`` does. Only the way ``U_n^m(delta)`` is computed changes:
every ``r^n P_n^m(cos t) e^{i m phi}`` is a homogeneous polynomial in ``(x, y, z)``,
so it is built by the recurrences of the complex inner solid harmonics in Cartesian
coordinates instead of from spherical angles -- a rewrite of the same functions,
not a Cartesian expansion. The harmonics are
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

# __all__ is at the end: lines above the kernels are their Triton IR locations


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


# ------------------------------------------------------------------ reverse
# The adjoint of the particle-major L2P (plan S8). Everything below the forward:
# the forward kernel's source lines are its Triton IR's locations, which the
# program-identity tool reads (bench/provider_hlo_identity.py).


def l2p_real_grad_cartesian_transpose(
    bx: Any, by: Any, bz: Any, x: Any, y: Any, z: Any, *, order: int
) -> list:
    """The coefficient cotangents of :func:`l2p_real_grad_cartesian`.

    The gradient is linear in the coefficients, ``g = sum_k F_k grad U_k(delta)``,
    so the cotangent of ``F_k`` for a cotangent ``b`` of ``g`` is
    ``b . grad U_k(delta)``: the same recurrence, every term contracted with
    ``b`` instead of summed with ``F``.

    Parameters
    ----------
    bx : Any
        Cotangent of ``gx`` (broadcastable to ``x``).
    by : Any
        Cotangent of ``gy``.
    bz : Any
        Cotangent of ``gz``.
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
    list
        ``(p+1)^2`` arrays in packed order
        (:func:`~jaccpot.operators._sh_indexing.sh_index`); the monopole's is zero
        (the gradient does not read it).
    """
    p = int(order)
    zero = x * 0
    out = [zero] * ((p + 1) ** 2)
    if p < 1:
        return out
    r2 = x * x + y * y + z * z
    Y = {0: (zero + 1, zero)}
    prev: dict = {}

    def add(*terms: Any) -> Any:
        acc = None
        for t in terms:
            if t is not None:
                acc = t if acc is None else acc + t
        return acc

    for n in range(1, p + 1):
        d = n - 1
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
            zr, zi = Y[m] if m <= d else (None, None)
            ar, ai = Y[m + 1] if m + 1 <= d else (None, None)
            if m >= 1:
                br, bi = Y[m - 1]
            elif d >= 1:
                br, bi = -Y[1][0], Y[1][1]
            else:
                br, bi = None, None
            # the forward's d_x = (B - A) / 2, d_y = i (A + B) / 2, d_z = Y_{n-1}^m
            sr = add(br, None if ar is None else -ar)
            si = add(bi, None if ai is None else -ai)
            tr = add(ar, br)
            ti = add(ai, bi)
            out[sh_index(n, m)] = add(
                None if zr is None else bz * zr,
                None if sr is None else bx * (0.5 * sr),
                None if ti is None else -(by * (0.5 * ti)),
            )
            if m > 0:
                out[sh_index(n, -m)] = add(
                    None if zi is None else bz * zi,
                    None if si is None else bx * (0.5 * si),
                    None if tr is None else by * (0.5 * tr),
                )
    return out


def _l2p_lane_cotangents(
    coeff: list,
    x: Array,
    y: Array,
    z: Array,
    bx: Array,
    by: Array,
    bz: Array,
    *,
    order: int,
) -> tuple[list, tuple]:
    """One lane's cotangents of the L2P gradient: of its coefficients and of ``delta``.

    The coefficient cotangents are :func:`l2p_real_grad_cartesian_transpose`. The
    ``delta`` cotangent is the Hessian of the expansion applied to ``b``: the
    gradient is the gradient of a scalar, so its Jacobian is symmetric and
    ``J^T b = J b``, ONE forward-mode derivative of the recurrence along ``b``.

    Parameters
    ----------
    coeff : list
        The lane's ``(p+1)^2`` coefficients, packed order.
    x : Array
        ``delta_x``.
    y : Array
        ``delta_y``.
    z : Array
        ``delta_z``.
    bx : Array
        Cotangent of the gradient's ``x`` component, in ``x``'s dtype.
    by : Array
        Of its ``y`` component.
    bz : Array
        Of its ``z`` component.
    order : int
        Expansion order. Static.

    Returns
    -------
    tuple[list, tuple]
        ``(coefficient cotangents [(p+1)^2], (delta_x, delta_y, delta_z) cotangent)``.
    """
    fbar = l2p_real_grad_cartesian_transpose(bx, by, bz, x, y, z, order=order)
    _, dbar = jax.jvp(
        lambda a, b, c: l2p_real_grad_cartesian(coeff, a, b, c, order=order),
        (x, y, z),
        (bx, by, bz),
    )
    return fbar, dbar


#: Leaves per program, lanes per leaf and warps of the reverse kernel. Timed alone on
#: an A100 at 8e6 (cell-like leaves of 1-37, p5, a shared card): 32 / 4 / 4 3.94 ms,
#: 16 / 4 / 2 3.98, 16 / 8 / 4 4.34, 8 / 8 / 2 4.38, against 1.15 ms for the forward
#: and 9.9-25 ms for a particle-major reverse (one particle per lane, the leaves'
#: sums by masked lane reductions across the program, an owner per leaf).
_REV_BLOCK, _REV_CHUNK, _REV_WARPS = 32, 4, 4


def _l2p_rev_block_kernel(
    pos_ref: KernelRef,
    cent_ref: KernelRef,
    coef_ref: KernelRef,
    leaf_node_ref: KernelRef,
    leaf_start_ref: KernelRef,
    leaf_count_ref: KernelRef,
    gbar_ref: KernelRef,
    pos_bar_in: KernelRef,
    cent_bar_in: KernelRef,
    coef_bar_in: KernelRef,
    pos_bar_ref: KernelRef,
    cent_bar_ref: KernelRef,
    coef_bar_ref: KernelRef,
    *,
    order: int,
    block: int,
    chunk: int,
    coeff_pad: int,
    rows: int,
    num_leaves: int,
) -> None:
    """``block`` leaves, their particles ``chunk`` lanes at a time (the P2M forward's shape).

    A program owns its leaves as the rows of a ``(block, chunk)`` lane tile and
    loops over chunks up to its largest leaf: each lane evaluates its particle's
    cotangents in its leaf's expansion (:func:`_l2p_lane_cotangents`) and writes
    its position cotangent; the leaf's coefficient and centre cotangents are
    summed along the row (``chunk`` lanes) and carried in registers across chunks,
    then written once, by this program -- no atomics. The leaf's
    coefficients are loaded once per program, not once per particle.

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
    leaf_start_ref : KernelRef
        Whole ``[L]`` first particle of each leaf.
    leaf_count_ref : KernelRef
        Whole ``[L]`` particles of each leaf.
    gbar_ref : KernelRef
        Whole ``[N, 3]`` cotangent of the forward's gradient.
    pos_bar_in : KernelRef
        The zeroed ``[N, 3]`` position cotangent, aliased to ``pos_bar_ref``.
    cent_bar_in : KernelRef
        The zeroed ``[nodes, 3]`` centre cotangent, aliased to ``cent_bar_ref``.
    coef_bar_in : KernelRef
        The zeroed ``[nodes, (p+1)^2]`` coefficient cotangent, aliased to
        ``coef_bar_ref``.
    pos_bar_ref : KernelRef
        **Output** position cotangent rows.
    cent_bar_ref : KernelRef
        **Output** centre cotangent rows of the program's leaves' nodes.
    coef_bar_ref : KernelRef
        **Output** coefficient cotangent rows of the program's leaves' nodes.
    order : int
        Expansion order. Static.
    block : int
        Leaves per program (a power of two). Static.
    chunk : int
        Lanes per leaf per iteration (a power of two). Static.
    coeff_pad : int
        Lanes of a coefficient row (a power of two >= ``(p+1)^2``). Static.
    rows : int
        ``N``. Static.
    num_leaves : int
        ``L``. Static.

    Returns
    -------
    None
        Writes the leaves' particle rows and the leaves' rows.
    """
    del pos_bar_in, cent_bar_in, coef_bar_in
    p = int(order)
    C = (p + 1) ** 2
    leaf = pl.program_id(0) * block + lax.broadcasted_iota(jnp.int32, (block,), 0)
    live = leaf < num_leaves
    lf = jnp.where(live, leaf, 0)
    nd = leaf_node_ref[lf]
    start = leaf_start_ref[lf]
    count = jnp.where(live, leaf_count_ref[lf], 0)
    cx = cent_ref[nd, 0][:, None]
    cy = cent_ref[nd, 1][:, None]
    cz = cent_ref[nd, 2][:, None]
    coeff = [coef_ref[nd, k][:, None] for k in range(C)]
    lane = lax.broadcasted_iota(jnp.int32, (block, chunk), 1)
    dtype = pos_bar_ref.dtype
    acc_dtype = coef_ref.dtype

    def body(t: Array, acc: tuple) -> tuple:
        off = t * chunk + lane
        valid = off < count[:, None]
        idx = jnp.where(valid, start[:, None] + off, 0)
        x = cx - pos_ref[idx, 0]
        y = cy - pos_ref[idx, 1]
        z = cz - pos_ref[idx, 2]
        bx = gbar_ref[idx, 0].astype(x.dtype)
        by = gbar_ref[idx, 1].astype(x.dtype)
        bz = gbar_ref[idx, 2].astype(x.dtype)
        fbar, (dx, dy, dz) = _l2p_lane_cotangents(coeff, x, y, z, bx, by, bz, order=p)
        # delta = centre - position: the position gets minus delta's cotangent
        dest = jnp.where(valid, idx, rows)
        plgpu.store(pos_bar_ref.at[dest, 0], (-dx).astype(dtype), mask=valid)
        plgpu.store(pos_bar_ref.at[dest, 1], (-dy).astype(dtype), mask=valid)
        plgpu.store(pos_bar_ref.at[dest, 2], (-dz).astype(dtype), mask=valid)
        # the C coefficient cotangents, then the centre's three, summed per leaf
        channels = list(fbar) + [dx, dy, dz]
        return tuple(
            a + jnp.sum(jnp.where(valid, c, 0), axis=1) for a, c in zip(acc, channels)
        )

    trips = (jnp.max(count) + (chunk - 1)) // chunk
    zero = jnp.zeros((block,), acc_dtype)
    acc = lax.fori_loop(0, trips, body, tuple(zero for _ in range(C + 3)))
    n_rows = int(coef_bar_ref.shape[0])
    row = jnp.where(live, nd, n_rows)[:, None]
    cidx = lax.broadcasted_iota(jnp.int32, (block, coeff_pad), 1)
    tile = jnp.zeros((block, coeff_pad), acc_dtype)
    for c in range(C):
        tile = jnp.where(cidx == c, acc[c][:, None], tile)
    plgpu.store(
        coef_bar_ref.at[row, cidx],
        tile.astype(coef_bar_ref.dtype),
        mask=live[:, None] & (cidx < C),
    )
    cidx3 = lax.broadcasted_iota(jnp.int32, (block, 4), 1)
    tile3 = jnp.zeros((block, 4), acc_dtype)
    for a in range(3):
        tile3 = jnp.where(cidx3 == a, acc[C + a][:, None], tile3)
    plgpu.store(
        cent_bar_ref.at[row, cidx3],
        tile3.astype(cent_bar_ref.dtype),
        mask=live[:, None] & (cidx3 < 3),
    )


def l2p_real_particles_reverse_pallas(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    grad_bar: Array,
    *,
    order: int,
    block: int = _REV_BLOCK,
    chunk: int = _REV_CHUNK,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = _REV_WARPS,
) -> tuple[Array, Array, Array]:
    """Adjoint of :func:`l2p_real_particles_pallas`, the leaves' sums in the kernel.

    One program per ``block`` leaves (:func:`_l2p_rev_block_kernel`): each
    particle's position cotangent (the expansion's Hessian applied to the
    cotangent) is written to its own row, and each leaf's coefficient and centre
    cotangents are summed over its particles in the program that owns the leaf,
    so the only arrays are the forward's own shapes: no per-particle coefficient
    cotangent (``N x (p+1)^2``, 19.6 GB at 1e8 in float32) exists, and no atomics.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]`` real local coefficients (the forward's).
    centers : Array
        ``[nodes, 3]`` expansion centres.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    leaf_nodes : Array
        ``[L]`` node id of each leaf (as the forward; the leaves' order is free).
    leaf_start : Array
        ``[L]`` first particle of each leaf.
    leaf_count : Array
        ``[L]`` particles of each leaf (0: empty).
    grad_bar : Array
        ``[N, 3]`` cotangent of the forward's gradient.
    order : int
        Expansion order. Static.
    block : int
        Leaves per program (rounded up to a power of two). Static.
    chunk : int
        Lanes per leaf per iteration (rounded up to a power of two). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(coefficients_bar [nodes, (p+1)^2], centers_bar [nodes, 3],
        positions_bar [N, 3])``, each in its primal's dtype; zero on the internal
        nodes, the empty leaves and the rows past the leaves' particles.

    Raises
    ------
    RuntimeError
        If Pallas is not available.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    p = int(order)
    C = (p + 1) ** 2
    n = int(positions_sorted.shape[0])
    L = int(leaf_nodes.shape[0])
    B = _next_pow2(block)
    pos = jnp.asarray(positions_sorted)
    coef = jnp.asarray(coefficients)
    cent = jnp.asarray(centers, coef.dtype)
    outs = [
        jnp.zeros((n, 3), pos.dtype),
        jnp.zeros(cent.shape, cent.dtype),
        jnp.zeros(coef.shape, coef.dtype),
    ]
    if n == 0 or L == 0:
        return outs[2], outs[1].astype(jnp.asarray(centers).dtype), outs[0]
    idx = jnp.asarray(leaf_nodes).dtype
    operands = [
        pos,
        cent,
        coef,
        jnp.asarray(leaf_nodes, idx),
        jnp.asarray(leaf_start, idx),
        jnp.maximum(jnp.asarray(leaf_count, idx), 0),
        jnp.asarray(grad_bar, pos.dtype),
    ]
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    pos_bar, cent_bar, coef_bar = pl.pallas_call(
        functools.partial(
            _l2p_rev_block_kernel,
            order=p,
            block=B,
            chunk=_next_pow2(chunk),
            coeff_pad=_next_pow2(C),
            rows=n,
            num_leaves=L,
        ),
        grid=(-(-L // B),),
        in_specs=[_full(o) for o in operands + outs],
        out_specs=[_full(o) for o in outs],
        out_shape=[jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outs],
        input_output_aliases={len(operands) + k: k for k in range(len(outs))},
        interpret=bool(interpret),
        name=f"l2p_rev_real_block_p{p}_b{B}",
        **backend_kwargs,
    )(*operands, *outs)
    return coef_bar, cent_bar.astype(jnp.asarray(centers).dtype), pos_bar


__all__ = [
    "l2p_real_grad_cartesian",
    "l2p_real_grad_cartesian_transpose",
    "l2p_real_particles_pallas",
    "l2p_real_particles_reverse_pallas",
]
