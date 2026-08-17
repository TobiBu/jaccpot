"""Mutual (double-sided) near-field P2P Pallas kernel.

Every other near-field Pallas lane in this tree is a **gather**: one program owns
a target leaf and streams its neighbours, summing into that target's registers.
That shape cannot express the mutual kernel, which owns one *leaf pair*,
evaluates the ``S x S`` block **once**, and emits two results -- ``+F`` for the
``a`` leaf's particles and ``-F`` for the ``b`` leaf's.

Why the two-sided emission has to be a negation
-----------------------------------------------
The whole point of :mod:`jaccpot.mutual` is that the momentum residual sits at
round-off (~1e-17 in float64) instead of at the force accuracy (~1e-3). That
holds only because the ``b`` side is the *bitwise negation* of the same
``scale * dr`` tile the ``a`` side reduced -- IEEE guarantees
``fl(x_b - x_a) == -fl(x_a - x_b)`` and the prefactor ``G m_i m_j w_ij r^-3`` is
symmetric to the last bit, so the two reductions differ only in summation order.
A port that recomputed ``dr`` for the ``b`` pass would still produce forces that
look right to every accuracy test and would silently move the momentum residual
up by fourteen orders of magnitude. Hence: one ``dx/dy/dz`` tile, two reductions
over opposite axes, and the sign applied on the way out.

The level weight ``level_weights[max(rung_i, rung_j)]`` is applied **inside** the
kernel, as one symmetric scalar per pair, before the tile is reduced. That is
what keeps momentum exact under *any* weighting: the same float multiplies ``+f``
and ``-f``.

HARDWARE: needs Ampere (sm_80+) for the Triton lowering; ``interpret=True`` runs
the identical kernel logic under CPU semantics, which is how the parity tests
stay non-vacuous without a GPU.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None

__all__ = [
    "pallas_nearfield_mutual_supported",
    "mutual_leafpair_block_jax",
    "mutual_leafpair_block_pallas",
    "mutual_leafpair_block_vjp_pallas",
    "mutual_leafpair_block_cvjp",
]

# Positions/forces are carried in width-4 lanes for aligned vector loads; lane 3
# is inert. Triton wants power-of-two tile dims, which 3 is not.
_VEC_WIDTH = 4


def pallas_nearfield_mutual_supported() -> bool:
    """True only on a GPU with compute capability >= 8.0 (Ampere+).

    Matches the gate the fused M2L kernel uses. A plain ``gpu`` check is not
    enough: on a pre-Ampere card the Triton lowering fails at runtime, so callers
    would get a crash rather than the pure-JAX fallback they asked for.
    """
    if pl is None:
        return False
    try:
        device = jax.devices()[0]
    except Exception:  # pragma: no cover - backend discovery is environment-dependent
        return False
    if device.platform != "gpu":
        return False
    capability = getattr(device, "compute_capability", None)
    if capability is None:
        return False
    try:
        return int(str(capability).split(".")[0]) >= 8
    except Exception:  # pragma: no cover
        return False


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _pair_weight_tile(
    rung_a_f: Array, rung_b_f: Array, level_weights: Array, num_levels: int
) -> Array:
    """``level_weights[max(rung_i, rung_j)]`` as an ``(S, S)`` tile.

    Built as a masked reduction over a ``(K, S, S)`` stack rather than either a
    gather or a Python loop of scalar table reads. Both alternatives are dead
    ends in the Triton lowering: a dynamic ``take`` lowers to ``gather``, and
    ``level_weights[k]`` with a *static* ``k`` lowers to ``slice`` — neither of
    which Pallas GPU implements. Only whole-array ops survive, so the table is
    reshaped to ``(K, 1, 1)`` and broadcast.

    ``K`` is ``k_max + 1``, typically <= 8, so the extra axis is cheap.

    Rungs arrive as floats (see the ``_f`` convention on the ``custom_vjp``
    boundary below); they are small exact integers, so the comparison is exact.
    """
    levels = int(num_levels)
    pair_level = jnp.maximum(rung_a_f[:, None], rung_b_f[None, :])
    table = jnp.reshape(level_weights, (levels, 1, 1))
    ladder = lax.broadcasted_iota(
        pair_level.dtype, (levels,) + tuple(pair_level.shape), 0
    )
    hit = jnp.abs(pair_level[None, ...] - ladder) < 0.5
    return jnp.sum(table * hit.astype(table.dtype), axis=0)


def _block_tile(
    a_xyz: tuple[Array, Array, Array],
    ma: Array,
    va_f: Array,
    b_xyz: tuple[Array, Array, Array],
    mb: Array,
    vb_f: Array,
    weight: Optional[Array],
    softening_sq: Array,
    g_value: Array,
    *,
    exclude_diagonal: bool,
) -> tuple[Array, Array, Array]:
    """Return the ``(S, S)`` per-component force tiles for one leaf pair.

    ``dr = x_b[j] - x_a[i]``, so the returned tiles are the force on ``a_i`` due
    to ``b_j``. The ``b`` side is obtained by the caller as the negated reduction
    over the other axis -- never recomputed.

    Coordinates arrive as three separate ``(S,)`` component vectors rather than
    one ``(S, 3)`` array: inside a Pallas kernel, slicing a component out of a
    packed array lowers to a ``gather``, which the Triton backend rejects
    outright ("Unimplemented primitive in Pallas GPU lowering: gather"). Pulling
    each component straight off its ref keeps every access a strided load.
    """
    ax, ay, az = a_xyz
    bx, by, bz = b_xyz
    dx = bx[None, :] - ax[:, None]
    dy = by[None, :] - ay[:, None]
    dz = bz[None, :] - az[:, None]
    r2 = dx * dx + dy * dy + dz * dz + softening_sq

    valid = (va_f[:, None] > 0.5) & (vb_f[None, :] > 0.5)
    if exclude_diagonal:
        rows = lax.broadcasted_iota(jnp.int32, valid.shape, 0)
        cols = lax.broadcasted_iota(jnp.int32, valid.shape, 1)
        valid = valid & (rows != cols)

    # Double-`where`: the guarded r2 is substituted *before* the reciprocal, so
    # neither the forward nor the analytic reverse ever evaluates r^-3 (or its
    # derivative) at a masked pair. A trailing `where` alone would still let a
    # NaN through the cotangent.
    safe_r2 = jnp.where(valid, r2, jnp.ones_like(r2))
    inv_r = lax.rsqrt(safe_r2)
    inv_r = jnp.where(valid, inv_r, jnp.zeros_like(inv_r))
    inv_r3 = inv_r * inv_r * inv_r

    scale = g_value * ma[:, None] * mb[None, :] * inv_r3
    if weight is not None:
        scale = scale * weight
    return scale * dx, scale * dy, scale * dz


def _mutual_leafpair_kernel(
    xa_ref: KernelRef,
    ma_ref: KernelRef,
    va_ref: KernelRef,
    xb_ref: KernelRef,
    mb_ref: KernelRef,
    vb_ref: KernelRef,
    ra_ref: KernelRef,
    rb_ref: KernelRef,
    lw_ref: KernelRef,
    soft_ref: KernelRef,
    g_ref: KernelRef,
    fa_ref: KernelRef,
    fb_ref: KernelRef,
    *,
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
) -> None:
    """One leaf pair: evaluate the block once, emit ``+F`` and ``-F``."""
    # `lw_ref[...]` materialises the whole (few-entry) table once; indexing the
    # ref per level would emit a load the Triton lowering turns into a gather.
    weight = (
        None
        if num_levels <= 0
        else _pair_weight_tile(ra_ref[0], rb_ref[0], lw_ref[...], num_levels)
    )
    fx, fy, fz = _block_tile(
        (xa_ref[0, :, 0], xa_ref[0, :, 1], xa_ref[0, :, 2]),
        ma_ref[0],
        va_ref[0],
        (xb_ref[0, :, 0], xb_ref[0, :, 1], xb_ref[0, :, 2]),
        mb_ref[0],
        vb_ref[0],
        weight,
        soft_ref[0],
        g_ref[0],
        exclude_diagonal=exclude_diagonal,
    )
    zero = jnp.zeros_like(ma_ref[0])
    fa_ref[0, :, 0] = jnp.sum(fx, axis=1)
    fa_ref[0, :, 1] = jnp.sum(fy, axis=1)
    fa_ref[0, :, 2] = jnp.sum(fz, axis=1)
    fa_ref[0, :, 3] = zero
    if emit_b:
        # The negation, not a recomputation. See the module docstring.
        fb_ref[0, :, 0] = -jnp.sum(fx, axis=0)
        fb_ref[0, :, 1] = -jnp.sum(fy, axis=0)
        fb_ref[0, :, 2] = -jnp.sum(fz, axis=0)
        fb_ref[0, :, 3] = zero
    else:
        fb_ref[0, :, 0] = zero
        fb_ref[0, :, 1] = zero
        fb_ref[0, :, 2] = zero
        fb_ref[0, :, 3] = zero


def _block_vjp_tiles(
    a_xyz: tuple[Array, Array, Array],
    ma: Array,
    va_f: Array,
    b_xyz: tuple[Array, Array, Array],
    mb: Array,
    vb_f: Array,
    weight: Optional[Array],
    softening_sq: Array,
    g_value: Array,
    fa_bar_xyz: tuple[Array, Array, Array],
    fb_bar_xyz: tuple[Array, Array, Array],
    *,
    exclude_diagonal: bool,
) -> tuple[tuple[Array, Array, Array], Array, tuple[Array, Array, Array], Array]:
    """Analytic reverse of :func:`_block_tile` plus its two reductions.

    Forward, per pair ``(i, j)``::

        s_ij = G w_ij m_i m_j r_ij^-3,   f_ij = s_ij * dr_ij
        F_a[i] = sum_j f_ij              F_b[j] = -sum_i f_ij

    so each pair's cotangent is fed from **both** endpoints::

        fbar_ij = Fbar_a[i] - Fbar_b[j]

    That difference is the one structural change from the gather-shaped rule in
    ``near_field.py::_leafpair_accel_analytic_vjp``, which only ever sees the
    target's cotangent. Differentiating ``f = s(r) dr`` then gives::

        drbar_ij = s_ij fbar_ij - 3 (s_ij / r_ij^2) (dr_ij . fbar_ij) dr_ij
        xbar_b[j] = sum_i drbar_ij       xbar_a[i] = -sum_j drbar_ij
        mbar_a[i] = sum_j (s_ij / m_i)(dr_ij . fbar_ij)

    The mass cotangent is written as ``G w m_j r^-3`` rather than ``s / m_i`` so
    a zero-mass padding slot cannot divide by zero.

    Components are passed and returned separately for the same reason as in
    :func:`_block_tile`: packed ``(S, 3)`` slicing lowers to a ``gather``, which
    the Pallas Triton backend does not implement.
    """
    ax, ay, az = a_xyz
    bx_pos, by_pos, bz_pos = b_xyz
    dx = bx_pos[None, :] - ax[:, None]
    dy = by_pos[None, :] - ay[:, None]
    dz = bz_pos[None, :] - az[:, None]
    r2 = dx * dx + dy * dy + dz * dz + softening_sq

    valid = (va_f[:, None] > 0.5) & (vb_f[None, :] > 0.5)
    if exclude_diagonal:
        rows = lax.broadcasted_iota(jnp.int32, valid.shape, 0)
        cols = lax.broadcasted_iota(jnp.int32, valid.shape, 1)
        valid = valid & (rows != cols)

    safe_r2 = jnp.where(valid, r2, jnp.ones_like(r2))
    inv_r = lax.rsqrt(safe_r2)
    inv_r = jnp.where(valid, inv_r, jnp.zeros_like(inv_r))
    inv_r3 = inv_r * inv_r * inv_r

    base = g_value * inv_r3
    if weight is not None:
        base = base * weight
    scale = base * ma[:, None] * mb[None, :]

    # Both endpoints of the pair contributed to the forward, so both feed back.
    fax, fay, faz = fa_bar_xyz
    fbx, fby, fbz = fb_bar_xyz
    bx = fax[:, None] - fbx[None, :]
    by = fay[:, None] - fby[None, :]
    bz = faz[:, None] - fbz[None, :]

    dot = dx * bx + dy * by + dz * bz
    radial = jnp.where(valid, -3.0 * scale * dot / safe_r2, jnp.zeros_like(dot))

    drbar_x = scale * bx + radial * dx
    drbar_y = scale * by + radial * dy
    drbar_z = scale * bz + radial * dz

    xa_bar = (
        -jnp.sum(drbar_x, axis=1),
        -jnp.sum(drbar_y, axis=1),
        -jnp.sum(drbar_z, axis=1),
    )
    xb_bar = (
        jnp.sum(drbar_x, axis=0),
        jnp.sum(drbar_y, axis=0),
        jnp.sum(drbar_z, axis=0),
    )
    ma_bar = jnp.sum(base * mb[None, :] * dot, axis=1)
    mb_bar = jnp.sum(base * ma[:, None] * dot, axis=0)
    return xa_bar, ma_bar, xb_bar, mb_bar


def _mutual_leafpair_vjp_kernel(
    xa_ref: KernelRef,
    ma_ref: KernelRef,
    va_ref: KernelRef,
    xb_ref: KernelRef,
    mb_ref: KernelRef,
    vb_ref: KernelRef,
    ra_ref: KernelRef,
    rb_ref: KernelRef,
    lw_ref: KernelRef,
    soft_ref: KernelRef,
    g_ref: KernelRef,
    fa_bar_ref: KernelRef,
    fb_bar_ref: KernelRef,
    xa_bar_ref: KernelRef,
    ma_bar_ref: KernelRef,
    xb_bar_ref: KernelRef,
    mb_bar_ref: KernelRef,
    *,
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
) -> None:
    """One leaf pair's analytic reverse, tile-bounded like the forward."""
    weight = (
        None
        if num_levels <= 0
        else _pair_weight_tile(ra_ref[0], rb_ref[0], lw_ref[...], num_levels)
    )
    fa_bar_xyz = (fa_bar_ref[0, :, 0], fa_bar_ref[0, :, 1], fa_bar_ref[0, :, 2])
    if emit_b:
        fb_bar_xyz = (fb_bar_ref[0, :, 0], fb_bar_ref[0, :, 1], fb_bar_ref[0, :, 2])
    else:
        # The forward wrote zeros to the b side, so it carries no cotangent.
        zeros = jnp.zeros_like(fa_bar_xyz[0])
        fb_bar_xyz = (zeros, zeros, zeros)
    xa_bar, ma_bar, xb_bar, mb_bar = _block_vjp_tiles(
        (xa_ref[0, :, 0], xa_ref[0, :, 1], xa_ref[0, :, 2]),
        ma_ref[0],
        va_ref[0],
        (xb_ref[0, :, 0], xb_ref[0, :, 1], xb_ref[0, :, 2]),
        mb_ref[0],
        vb_ref[0],
        weight,
        soft_ref[0],
        g_ref[0],
        fa_bar_xyz,
        fb_bar_xyz,
        exclude_diagonal=exclude_diagonal,
    )
    zero = jnp.zeros_like(ma_bar)
    for lane in range(3):
        xa_bar_ref[0, :, lane] = xa_bar[lane]
        xb_bar_ref[0, :, lane] = xb_bar[lane]
    xa_bar_ref[0, :, 3] = zero
    xb_bar_ref[0, :, 3] = zero
    ma_bar_ref[0, :] = ma_bar
    mb_bar_ref[0, :] = mb_bar


def _pad_inputs(
    xa: Array, ma: Array, va_f: Array, rung_a_f: Optional[Array], width: int, dtype: Any
) -> tuple[Array, Array, Array, Array]:
    """Pad a leaf block's slot axis out to the Triton tile width.

    Padded slots get mask 0, so they are inert in every tile the kernels build.
    """
    slots = int(ma.shape[1])
    pad = width - slots
    xa_p = jnp.pad(xa, ((0, 0), (0, pad), (0, _VEC_WIDTH - 3))).astype(dtype)
    ma_p = jnp.pad(ma, ((0, 0), (0, pad))).astype(dtype)
    va_p = jnp.pad(jnp.asarray(va_f, dtype=dtype), ((0, 0), (0, pad)))
    ra_p = (
        jnp.zeros((int(ma.shape[0]), width), dtype=dtype)
        if rung_a_f is None
        else jnp.pad(jnp.asarray(rung_a_f, dtype=dtype), ((0, 0), (0, pad)))
    )
    return xa_p, ma_p, va_p, ra_p


def mutual_leafpair_block_jax(
    xa: Array,
    ma: Array,
    va_f: Array,
    xb: Array,
    mb: Array,
    vb_f: Array,
    rung_a_f: Optional[Array],
    rung_b_f: Optional[Array],
    level_weights: Optional[Array],
    softening_sq: Array,
    g_value: Array,
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
) -> tuple[Array, Array]:
    """Pure-jnp twin of the kernel: the correctness and AD oracle.

    Returns ``(F_a, F_b)`` of shape ``(pairs, slots, 3)``.
    """
    num_levels = 0 if level_weights is None else int(level_weights.shape[0])

    def one(
        xa_i: Array,
        ma_i: Array,
        va_i: Array,
        xb_i: Array,
        mb_i: Array,
        vb_i: Array,
        ra_i: Array,
        rb_i: Array,
    ) -> tuple[Array, Array]:
        weight = (
            None
            if num_levels <= 0
            else _pair_weight_tile(ra_i, rb_i, level_weights, num_levels)
        )
        fx, fy, fz = _block_tile(
            (xa_i[:, 0], xa_i[:, 1], xa_i[:, 2]),
            ma_i,
            va_i,
            (xb_i[:, 0], xb_i[:, 1], xb_i[:, 2]),
            mb_i,
            vb_i,
            weight,
            softening_sq,
            g_value,
            exclude_diagonal=exclude_diagonal,
        )
        f_a = jnp.stack(
            [jnp.sum(fx, axis=1), jnp.sum(fy, axis=1), jnp.sum(fz, axis=1)], axis=-1
        )
        if emit_b:
            f_b = -jnp.stack(
                [jnp.sum(fx, axis=0), jnp.sum(fy, axis=0), jnp.sum(fz, axis=0)], axis=-1
            )
        else:
            f_b = jnp.zeros_like(f_a)
        return f_a, f_b

    zeros = jnp.zeros(ma.shape, dtype=ma.dtype)
    return jax.vmap(one)(
        xa,
        ma,
        jnp.asarray(va_f, dtype=ma.dtype),
        xb,
        mb,
        jnp.asarray(vb_f, dtype=ma.dtype),
        zeros if rung_a_f is None else jnp.asarray(rung_a_f, dtype=ma.dtype),
        zeros if rung_b_f is None else jnp.asarray(rung_b_f, dtype=ma.dtype),
    )


def mutual_leafpair_block_pallas(
    xa: Array,
    ma: Array,
    va_f: Array,
    xb: Array,
    mb: Array,
    vb_f: Array,
    rung_a_f: Optional[Array],
    rung_b_f: Optional[Array],
    level_weights: Optional[Array],
    softening_sq: Array,
    g_value: Array,
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
    interpret: bool = False,
    backend: str = "triton",
) -> tuple[Array, Array]:
    """One Pallas program per leaf pair; same semantics as the jnp twin."""
    pairs, slots = int(ma.shape[0]), int(ma.shape[1])
    dtype = jnp.asarray(xa).dtype
    if pairs == 0:
        empty = jnp.zeros((pairs, slots, 3), dtype=dtype)
        return empty, empty

    width = _next_pow2(slots)
    num_levels = 0 if level_weights is None else int(level_weights.shape[0])
    lw = (
        jnp.ones((1,), dtype=dtype)
        if level_weights is None
        else jnp.asarray(level_weights, dtype=dtype)
    )

    xa_p, ma_p, va_p, ra_p = _pad_inputs(xa, ma, va_f, rung_a_f, width, dtype)
    xb_p, mb_p, vb_p, rb_p = _pad_inputs(xb, mb, vb_f, rung_b_f, width, dtype)
    soft = jnp.asarray(softening_sq, dtype=dtype).reshape((1,))
    gval = jnp.asarray(g_value, dtype=dtype).reshape((1,))

    kernel = functools.partial(
        _mutual_leafpair_kernel,
        num_levels=num_levels,
        exclude_diagonal=bool(exclude_diagonal),
        emit_b=bool(emit_b),
    )

    def bs_vec(cols: int) -> "pl.BlockSpec":
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_mat() -> "pl.BlockSpec":
        return pl.BlockSpec((1, width, _VEC_WIDTH), lambda i: (i, 0, 0))

    def bs_full(arr: Array) -> "pl.BlockSpec":
        shape = tuple(arr.shape)
        return pl.BlockSpec(shape, (lambda *_: (0,) * len(shape)))

    f_a, f_b = pl.pallas_call(
        kernel,
        grid=(pairs,),
        in_specs=[
            bs_mat(),
            bs_vec(width),
            bs_vec(width),
            bs_mat(),
            bs_vec(width),
            bs_vec(width),
            bs_vec(width),
            bs_vec(width),
            bs_full(lw),
            bs_full(soft),
            bs_full(gval),
        ],
        out_specs=[bs_mat(), bs_mat()],
        out_shape=[
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
        ],
        interpret=bool(interpret),
        **pallas_backend_kwargs(backend, interpret),
        name=f"mutual_leafpair_s{width}",
    )(xa_p, ma_p, va_p, xb_p, mb_p, vb_p, ra_p, rb_p, lw, soft, gval)
    return f_a[:, :slots, :3], f_b[:, :slots, :3]


def mutual_leafpair_block_vjp_pallas(
    xa: Array,
    ma: Array,
    va_f: Array,
    xb: Array,
    mb: Array,
    vb_f: Array,
    rung_a_f: Optional[Array],
    rung_b_f: Optional[Array],
    level_weights: Optional[Array],
    softening_sq: Array,
    g_value: Array,
    fa_bar: Array,
    fb_bar: Array,
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
    interpret: bool = False,
    backend: str = "triton",
) -> tuple[Array, Array, Array, Array]:
    """Hand-written analytic reverse, one Pallas program per leaf pair.

    Returns ``(xa_bar, ma_bar, xb_bar, mb_bar)``. Never itself differentiated, so
    its intermediates stay tile-bounded: the ``S x S`` tiles live in registers and
    the only HBM traffic is ``O(pairs * S)``, against the ``O(pairs * S^2)``
    transient that linearizing the pure-JAX twin materialises.
    """
    pairs, slots = int(ma.shape[0]), int(ma.shape[1])
    dtype = jnp.asarray(xa).dtype
    if pairs == 0:
        return (
            jnp.zeros((pairs, slots, 3), dtype=dtype),
            jnp.zeros((pairs, slots), dtype=dtype),
            jnp.zeros((pairs, slots, 3), dtype=dtype),
            jnp.zeros((pairs, slots), dtype=dtype),
        )

    width = _next_pow2(slots)
    num_levels = 0 if level_weights is None else int(level_weights.shape[0])
    lw = (
        jnp.ones((1,), dtype=dtype)
        if level_weights is None
        else jnp.asarray(level_weights, dtype=dtype)
    )

    xa_p, ma_p, va_p, ra_p = _pad_inputs(xa, ma, va_f, rung_a_f, width, dtype)
    xb_p, mb_p, vb_p, rb_p = _pad_inputs(xb, mb, vb_f, rung_b_f, width, dtype)
    pad = width - slots
    fa_bar_p = jnp.pad(fa_bar, ((0, 0), (0, pad), (0, _VEC_WIDTH - 3))).astype(dtype)
    fb_bar_p = jnp.pad(fb_bar, ((0, 0), (0, pad), (0, _VEC_WIDTH - 3))).astype(dtype)
    soft = jnp.asarray(softening_sq, dtype=dtype).reshape((1,))
    gval = jnp.asarray(g_value, dtype=dtype).reshape((1,))

    kernel = functools.partial(
        _mutual_leafpair_vjp_kernel,
        num_levels=num_levels,
        exclude_diagonal=bool(exclude_diagonal),
        emit_b=bool(emit_b),
    )

    def bs_vec(cols: int) -> "pl.BlockSpec":
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_mat() -> "pl.BlockSpec":
        return pl.BlockSpec((1, width, _VEC_WIDTH), lambda i: (i, 0, 0))

    def bs_full(arr: Array) -> "pl.BlockSpec":
        shape = tuple(arr.shape)
        return pl.BlockSpec(shape, (lambda *_: (0,) * len(shape)))

    xa_bar, ma_bar, xb_bar, mb_bar = pl.pallas_call(
        kernel,
        grid=(pairs,),
        in_specs=[
            bs_mat(),
            bs_vec(width),
            bs_vec(width),
            bs_mat(),
            bs_vec(width),
            bs_vec(width),
            bs_vec(width),
            bs_vec(width),
            bs_full(lw),
            bs_full(soft),
            bs_full(gval),
            bs_mat(),
            bs_mat(),
        ],
        out_specs=[bs_mat(), bs_vec(width), bs_mat(), bs_vec(width)],
        out_shape=[
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
            jax.ShapeDtypeStruct((pairs, width), dtype),
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
            jax.ShapeDtypeStruct((pairs, width), dtype),
        ],
        interpret=bool(interpret),
        **pallas_backend_kwargs(backend, interpret),
        name=f"mutual_leafpair_vjp_s{width}",
    )(
        xa_p,
        ma_p,
        va_p,
        xb_p,
        mb_p,
        vb_p,
        ra_p,
        rb_p,
        lw,
        soft,
        gval,
        fa_bar_p,
        fb_bar_p,
    )
    return (
        xa_bar[:, :slots, :3],
        ma_bar[:, :slots],
        xb_bar[:, :slots, :3],
        mb_bar[:, :slots],
    )


# ---------------------------------------------------------------------------
# Differentiable wrapper. ``pallas_call`` has no JVP/transpose rule, so the
# forward kernel is unusable under ``jax.grad`` without this. Per the PR-2
# post-mortem in ``docs/differentiable_fmm_pallas_vjp_plan.md``: the ``custom_vjp``
# is module-level (it closes over no tracer), the hashable statics go in
# ``nondiff_argnums``, and every ARRAY argument stays a regular positional
# argument -- including the non-differentiable index/mask arrays, which get
# ``jnp.zeros_like`` cotangents from ``bwd`` rather than a ``nondiff_argnums``
# slot that would try to hash a tracer.
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(11, 12, 13, 14))
def mutual_leafpair_block_cvjp(
    xa: Array,
    ma: Array,
    va_f: Array,
    xb: Array,
    mb: Array,
    vb_f: Array,
    rung_a_f: Array,
    rung_b_f: Array,
    level_weights: Array,
    softening_sq: Array,
    g_value: Array,
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
    interpret: bool,
) -> tuple[Array, Array]:
    """Differentiable mutual leaf-pair block: Pallas forward, analytic reverse."""
    return mutual_leafpair_block_pallas(
        xa,
        ma,
        va_f,
        xb,
        mb,
        vb_f,
        rung_a_f,
        rung_b_f,
        None if int(num_levels) <= 0 else level_weights,
        softening_sq,
        g_value,
        exclude_diagonal=bool(exclude_diagonal),
        emit_b=bool(emit_b),
        interpret=bool(interpret),
    )


def _mutual_leafpair_block_cvjp_fwd(
    xa: Array,
    ma: Array,
    va_f: Array,
    xb: Array,
    mb: Array,
    vb_f: Array,
    rung_a_f: Array,
    rung_b_f: Array,
    level_weights: Array,
    softening_sq: Array,
    g_value: Array,
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
    interpret: bool,
) -> tuple[tuple[Array, Array], tuple[Array, ...]]:
    out = mutual_leafpair_block_pallas(
        xa,
        ma,
        va_f,
        xb,
        mb,
        vb_f,
        rung_a_f,
        rung_b_f,
        None if int(num_levels) <= 0 else level_weights,
        softening_sq,
        g_value,
        exclude_diagonal=bool(exclude_diagonal),
        emit_b=bool(emit_b),
        interpret=bool(interpret),
    )
    # Residual is O(pairs * S), not O(pairs * S^2): the block tile is rebuilt in
    # the reverse kernel rather than stored. Under the caller's `lax.scan` these
    # residuals are stacked per chunk, so this is the term that decides whether a
    # galaxy-scale gradient fits in memory at all.
    residual = (
        xa,
        ma,
        va_f,
        xb,
        mb,
        vb_f,
        rung_a_f,
        rung_b_f,
        level_weights,
        softening_sq,
        g_value,
    )
    return out, residual


def _mutual_leafpair_block_cvjp_bwd(
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
    interpret: bool,
    residual: tuple[Array, ...],
    cotangent: tuple[Array, Array],
) -> tuple[Array, ...]:
    (
        xa,
        ma,
        va_f,
        xb,
        mb,
        vb_f,
        rung_a_f,
        rung_b_f,
        level_weights,
        softening_sq,
        g_value,
    ) = residual
    fa_bar, fb_bar = cotangent
    xa_bar, ma_bar, xb_bar, mb_bar = mutual_leafpair_block_vjp_pallas(
        xa,
        ma,
        va_f,
        xb,
        mb,
        vb_f,
        rung_a_f,
        rung_b_f,
        None if int(num_levels) <= 0 else level_weights,
        softening_sq,
        g_value,
        fa_bar,
        fb_bar,
        exclude_diagonal=bool(exclude_diagonal),
        emit_b=bool(emit_b),
        interpret=bool(interpret),
    )
    # Masks, rungs and the level table are discrete or frozen: they take zero
    # cotangents. They travel as regular FLOAT array args -- never
    # `nondiff_argnums`, which is for hashable statics, and never as bool/int
    # arrays, whose tangent type is `float0` and cannot take a `zeros_like`.
    return (
        xa_bar,
        ma_bar,
        jnp.zeros_like(va_f),
        xb_bar,
        mb_bar,
        jnp.zeros_like(vb_f),
        jnp.zeros_like(rung_a_f),
        jnp.zeros_like(rung_b_f),
        jnp.zeros_like(level_weights),
        jnp.zeros_like(softening_sq),
        jnp.zeros_like(g_value),
    )


mutual_leafpair_block_cvjp.defvjp(
    _mutual_leafpair_block_cvjp_fwd, _mutual_leafpair_block_cvjp_bwd
)
