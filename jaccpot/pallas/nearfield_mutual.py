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
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, jaxtyped

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

# THE SLOT AXIS IS `w`, AND THE PROSE IN THIS FILE CALLS IT `slots`.
#
# Same axis, two names, and the annotations use the package-wide one. These blocks
# are `leaf_particles[la]` gathered whole (see `mutual/nearfield.py::_pallas_block`),
# so the slot count IS the leaf width -- `w` in STYLE_GUIDE section 4.3, `max_leaf_size`
# in the config. The prose was not rewritten to match: `slots` is what the kernel
# calls its own tile dimension and appears in ~40 docstrings here, and renaming it
# would be a diff about vocabulary in a change that is about contracts.
#
# `w` IS SHARED BY THE `a` AND `b` SIDES, WHICH IS A REAL ASSERTION AND NOT A
# MEASURED COINCIDENCE. `_block_tile` itself broadcasts `a[:, None]` against
# `b[None, :]` and would accept unequal widths, so the pure-jnp twin alone would
# tolerate them. The Pallas path does not: `width` is `_next_pow2` of the *a*-side
# slot count and both sides are padded to it, so a wider `b` gives `jnp.pad` a
# negative width. Annotating the two sides `w` and `wb` was the alternative; it was
# rejected because the twin exists to mirror the kernel, and a caller relying on the
# twin accepting a shape the kernel rejects is relying on the two disagreeing.
# Measured: one problem size, `(6, 8)` -- see the coverage note on the decorators.

# Positions/forces are carried in width-4 lanes for aligned vector loads; lane 3
# is inert. Triton wants power-of-two tile dims, which 3 is not.
_VEC_WIDTH = 4


def pallas_nearfield_mutual_supported() -> bool:
    """True only on a GPU with compute capability >= 8.0 (Ampere+).

    Matches the gate the fused M2L kernel uses. A plain ``gpu`` check is not
    enough: on a pre-Ampere card the Triton lowering fails at runtime, so callers
    would get a crash rather than the pure-JAX fallback they asked for.

    Returns
    -------
    bool
        ``True`` when the fused mutual near-field kernel may be dispatched.
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
    """Round ``n`` up to a power of two.

    Triton tiles must be power-of-two sized, so a leaf block's slot axis is
    padded out to this width.

    Parameters
    ----------
    n : int
        Value to round up.

    Returns
    -------
    int
        Smallest power of two that is at least ``n``.
    """
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


# BARE ON PURPOSE, and it is the permissive end of the module: `bench/annotation_pilot`
# measures 33 of this file's 99 silent acceptances in these three helpers, including
# `_block_tile` accepting an extra leading axis on all eleven of its arrays.
#
# They are still bare because they are called from inside two `pallas_call` bodies
# (`_mutual_leafpair_kernel`, `_mutual_leafpair_vjp_kernel`) as well as from the
# vmapped twin, so a `@jaxtyped` here executes inside a Pallas kernel trace. That is
# the one lane this box cannot exercise -- interpret mode is not the Triton lowering,
# and there is no GPU leg in CI -- and `jaccpot/pallas/` has carried no enforced check
# until now. Shipping the first one into a kernel body on an argument-from-analogy
# would be the wrong order.
#
# What unblocks them: a `python -m bench.gpu_gate` run on an Ampere+ card, or a
# re-record of the two `*_pallas` entry points with `interpret=False`. Until then the
# entry points above carry the contract, and every call into these helpers comes
# through one of them.


# `slots` is the tile width the three helpers below share, and it is deliberately not the
# entry points' `w`. `_pad_inputs` rounds the slot count up to a power of two before the
# kernel sees it, so inside a tile the width is the PADDED one: measured at 8 where the
# contract's `w` was 4. Naming both `w` would assert an equality that does not hold, which
# is why `sw` exists in STYLE_GUIDE 4.3 for the same reason one lane over.
#
# What these three annotations are FOR, measured with `bench/annotation_pilot.py` on a
# re-record of this module (213 perturbations, 8 functions measured, 2 inconclusive):
#
#     _block_tile          12 array params   10 accepted
#     _block_vjp_tiles     18 array params   16 accepted
#     _pair_weight_tile     3 array params    5 accepted
#     the four entry points and `_pad_inputs`  0 accepted, except one -- see below
#
# All 31 acceptances are here, in the internals; fb2d0a1's boundary contract rejects
# everything thrown at it. 26 of the 31 are `extra leading axis`, which neither call site
# can deliver -- a Pallas ref slice and a `vmap` example are both rank-fixed. The five that
# matter are on `_pair_weight_tile`, where a rung array SHORTER than the tile is accepted:
# that is the "length mismatch between arrays that must agree" mode, and #297 is the
# precedent for it being reachable through a lane's internals even when the boundary is
# contracted -- there a BlockSpec read the target width from a source table and dropped
# real particles in silence.
#
# The one entry-point acceptance is NOT a gap a shape can close. `level_weights` shortened
# by one is accepted by `mutual_leafpair_block_jax` because `levels` binds only to that
# parameter, so it asserts nothing (4.4). A short table is a legitimate input; what makes
# it dangerous is that rungs INDEX it, and an out-of-range rung clamps. That is a
# value-length relation, not a shape one, and it is left as a finding rather than
# annotated away.
#
@jaxtyped(typechecker=beartype)
def _pair_weight_tile(
    rung_a_f: Float[Array, "slots"],
    rung_b_f: Float[Array, "slots"],
    level_weights: Float[Array, "levels"],
    num_levels: int,
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

    Parameters
    ----------
    rung_a_f : Float[Array, 'slots']
        ``(S,)`` leaf-A rung, float-encoded.
    rung_b_f : Float[Array, 'slots']
        ``(S,)`` leaf-B rung, float-encoded.
    level_weights : Float[Array, 'levels']
        ``(num_levels,)`` weight per interaction level, read whole.
    num_levels : int
        Number of levels, i.e. the ``K`` of the masked reduction. Static.

    Returns
    -------
    Array
        ``(S, S)`` tile of ``level_weights[max(rung_i, rung_j)]``.
    """
    levels = int(num_levels)
    table = jnp.reshape(level_weights, (levels, 1, 1))
    hit = _pair_level_hits(rung_a_f, rung_b_f, levels)
    return jnp.sum(table * hit.astype(table.dtype), axis=0)


def _pair_level_hits(rung_a_f: Array, rung_b_f: Array, levels: int) -> Array:
    """``(K, S, S)`` mask: does pair ``(i, j)`` sit on level ``k = max(rung_i, rung_j)``?

    The one-hot ladder :func:`_pair_weight_tile` contracts the weight table
    against, factored out so the reverse can contract the *unweighted* pair
    cotangent against the same ladder and get the per-level weight cotangent.
    Same Triton constraints, same construction: whole-array ops only.

    Parameters
    ----------
    rung_a_f : Array
        ``(S,)`` leaf-A rung, float-encoded.
    rung_b_f : Array
        ``(S,)`` leaf-B rung, float-encoded.
    levels : int
        Number of ladder rungs ``K``; a pair whose level is ``>= K`` hits nothing.

    Returns
    -------
    Array
        ``(K, S, S)`` boolean one-hot mask over the pair level.
    """
    pair_level = jnp.maximum(rung_a_f[:, None], rung_b_f[None, :])
    ladder = lax.broadcasted_iota(
        pair_level.dtype, (int(levels),) + tuple(pair_level.shape), 0
    )
    return jnp.abs(pair_level[None, ...] - ladder) < 0.5


@jaxtyped(typechecker=beartype)
def _block_tile(
    a_xyz: tuple[Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]],
    ma: Float[Array, "slots"],
    va_f: Float[Array, "slots"],
    b_xyz: tuple[Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]],
    mb: Float[Array, "slots"],
    vb_f: Float[Array, "slots"],
    weight: Optional[Float[Array, "slots slots"]],
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

    Parameters
    ----------
    a_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Leaf-A coordinates as three separate ``(S,)`` component vectors.
    ma : Float[Array, 'slots']
        ``(S,)`` leaf-A masses.
    va_f : Float[Array, 'slots']
        ``(S,)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    b_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Leaf-B coordinates, same layout as ``a_xyz``.
    mb : Float[Array, 'slots']
        ``(S,)`` leaf-B masses.
    vb_f : Float[Array, 'slots']
        ``(S,)`` leaf-B validity mask, same encoding.
    weight : Optional[Float[Array, 'slots slots']]
        ``(S, S)`` level weight from :func:`_pair_weight_tile`, or ``None`` for
        unit weights.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    exclude_diagonal : bool
        Drop ``i == j``, for a leaf paired with itself.

    Returns
    -------
    tuple[Array, Array, Array]
        The ``(S, S)`` x, y and z force tiles: the force on ``a_i`` due to
        ``b_j``. Masked pairs are exactly zero.
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
    """One leaf pair: evaluate the block once, emit ``+F`` and ``-F``.
    Evaluating once and emitting both signs is what makes momentum cancel
    algebraically rather than to within rounding.

    Parameters
    ----------
    xa_ref : KernelRef
        Leaf-A positions for this pair, ``(1, S, VEC)``.
    ma_ref : KernelRef
        Leaf-A masses, ``(1, S)``.
    va_ref : KernelRef
        Leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb_ref : KernelRef
        Leaf-B positions, same layout as ``xa_ref``.
    mb_ref : KernelRef
        Leaf-B masses.
    vb_ref : KernelRef
        Leaf-B validity mask, same encoding.
    ra_ref : KernelRef
        Leaf-A rungs, float-encoded.
    rb_ref : KernelRef
        Leaf-B rungs, float-encoded.
    lw_ref : KernelRef
        Level-weight table.
    soft_ref : KernelRef
        Squared softening length.
    g_ref : KernelRef
        Gravitational constant.
    fa_ref : KernelRef
        Output: force on the leaf-A particles.
    fb_ref : KernelRef
        Output: force on the leaf-B particles, written only when ``emit_b``.
    num_levels : int
        Number of interaction levels. ``<= 0`` disables level weighting entirely.
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.

    Returns
    -------
    None
        Writes through ``fa_ref`` and ``fb_ref``.
    """
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


@jaxtyped(typechecker=beartype)
def _block_vjp_tiles(
    a_xyz: tuple[Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]],
    ma: Float[Array, "slots"],
    va_f: Float[Array, "slots"],
    b_xyz: tuple[Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]],
    mb: Float[Array, "slots"],
    vb_f: Float[Array, "slots"],
    weight: Optional[Float[Array, "slots slots"]],
    softening_sq: Array,
    g_value: Array,
    fa_bar_xyz: tuple[
        Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]
    ],
    fb_bar_xyz: tuple[
        Float[Array, "slots"], Float[Array, "slots"], Float[Array, "slots"]
    ],
    *,
    exclude_diagonal: bool,
    level_hits: Optional[Bool[Array, "levels slots slots"]] = None,
) -> tuple[
    tuple[Array, Array, Array],
    Array,
    tuple[Array, Array, Array],
    Array,
    Optional[Array],
    Array,
    Array,
]:
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

    The three scalar parameters the force is smooth in get cotangents too. With
    ``p_ij = m_i m_j r_ij^-3`` and ``dot_ij = dr_ij . fbar_ij``::

        Gbar        = sum_ij w_ij p_ij dot_ij               (f is linear in G)
        wbar[k]     = sum_{ij on level k} G p_ij dot_ij     (f is linear in w_ij)
        softbar     = sum_ij -(3/2) (s_ij / r_ij^2) dot_ij  (r^2 = |dr|^2 + soft)

    ``softbar`` is half the sum of the ``radial`` tile that the position
    cotangent already needs, so it costs one reduction. The level cotangent
    contracts the unweighted pair cotangent against the same one-hot ladder the
    forward contracted the weight table against (``level_hits``), so it costs
    ``K`` masked reductions of a tile that is already in registers. Until these
    existed the reverse rule returned zeros for all three, which made
    ``d/d(dt_max)`` through the Pallas near field wrong by the near field's
    whole share -- measured AD/FD = 0.056 on a two-clump N = 256 system
    (jaccpot#316 pinned it) -- and silently dropped ``d/d(softening)`` and
    ``d/d(G)``.

    Components are passed and returned separately for the same reason as in
    :func:`_block_tile`: packed ``(S, 3)`` slicing lowers to a ``gather``, which
    the Pallas Triton backend does not implement.

    Parameters
    ----------
    a_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Leaf-A coordinates as three ``(S,)`` component vectors.
    ma : Float[Array, 'slots']
        ``(S,)`` leaf-A masses.
    va_f : Float[Array, 'slots']
        ``(S,)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    b_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Leaf-B coordinates, same layout.
    mb : Float[Array, 'slots']
        ``(S,)`` leaf-B masses.
    vb_f : Float[Array, 'slots']
        ``(S,)`` leaf-B validity mask, same encoding.
    weight : Optional[Float[Array, 'slots slots']]
        ``(S, S)`` level weight, or ``None`` for unit weights.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    fa_bar_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Cotangent of the leaf-A force, as three ``(S,)`` components.
    fb_bar_xyz : tuple[Float[Array, 'slots'], Float[Array, 'slots'], Float[Array, 'slots']]
        Cotangent of the leaf-B force, same layout. Both endpoints feed each
        pair, which is the structural difference from the gather-shaped rule.
    exclude_diagonal : bool
        Drop ``i == j``, for a leaf paired with itself.
    level_hits : Optional[Bool[Array, 'levels slots slots']]
        ``(K, S, S)`` one-hot level mask from :func:`_pair_level_hits`, required
        when ``weight`` is given; the level-weight cotangent is contracted
        against it. ``None`` with unit weights.

    Returns
    -------
    tuple[tuple[Array, Array, Array], Array, tuple[Array, Array, Array], Array, Optional[Array], Array, Array]
        ``(xa_bar_xyz, ma_bar, xb_bar_xyz, mb_bar, lw_bar, soft_bar, g_bar)`` --
        position cotangents as component triples, mass cotangents as ``(S,)``
        vectors, the ``(K,)`` level-weight cotangent (``None`` with unit
        weights), and the scalar softening-squared and ``G`` cotangents, each
        this leaf pair's contribution.
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

    # `pair` is m_i m_j r^-3; `base` is G w r^-3 (what the mass cotangents need);
    # `scale` is the full G w m_i m_j r^-3 prefactor of the forward.
    pair = inv_r3 * ma[:, None] * mb[None, :]
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

    # The scalar parameters. Reductions are done one axis at a time -- the same
    # shape of op `_pair_weight_tile` already lowers -- rather than as one
    # multi-axis reduce.
    unweighted = g_value * pair * dot  # d f_ij / d w_ij . fbar_ij
    if weight is not None:
        assert level_hits is not None, "level_hits is required with a weight tile"
        lw_bar = jnp.sum(
            jnp.sum(level_hits.astype(unweighted.dtype) * unweighted[None], axis=2),
            axis=1,
        )
        g_bar = jnp.sum(jnp.sum(pair * weight * dot, axis=1), axis=0)
    else:
        lw_bar = None
        g_bar = jnp.sum(jnp.sum(pair * dot, axis=1), axis=0)
    # d f_ij / d(soft) = -(3/2) s_ij dr_ij / r_ij^2, and `radial` is
    # -3 s_ij dot_ij / r_ij^2 (zero on masked pairs), so this is half its sum.
    soft_bar = 0.5 * jnp.sum(jnp.sum(radial, axis=1), axis=0)
    return xa_bar, ma_bar, xb_bar, mb_bar, lw_bar, soft_bar, g_bar


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
    lw_bar_ref: KernelRef,
    scalar_bar_ref: KernelRef,
    *,
    num_levels: int,
    levels_width: int,
    exclude_diagonal: bool,
    emit_b: bool,
) -> None:
    """One leaf pair's analytic reverse, tile-bounded like the forward.

    Parameters
    ----------
    xa_ref : KernelRef
        Leaf-A positions for this pair, ``(1, S, VEC)``.
    ma_ref : KernelRef
        Leaf-A masses, ``(1, S)``.
    va_ref : KernelRef
        Leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb_ref : KernelRef
        Leaf-B positions, same layout as ``xa_ref``.
    mb_ref : KernelRef
        Leaf-B masses.
    vb_ref : KernelRef
        Leaf-B validity mask, same encoding.
    ra_ref : KernelRef
        Leaf-A rungs, float-encoded.
    rb_ref : KernelRef
        Leaf-B rungs, float-encoded.
    lw_ref : KernelRef
        Level-weight table.
    soft_ref : KernelRef
        Squared softening length.
    g_ref : KernelRef
        Gravitational constant.
    fa_bar_ref : KernelRef
        Cotangent of the leaf-A force.
    fb_bar_ref : KernelRef
        Cotangent of the leaf-B force.
    xa_bar_ref : KernelRef
        Output: leaf-A position cotangent.
    ma_bar_ref : KernelRef
        Output: leaf-A mass cotangent.
    xb_bar_ref : KernelRef
        Output: leaf-B position cotangent.
    mb_bar_ref : KernelRef
        Output: leaf-B mass cotangent.
    lw_bar_ref : KernelRef
        Output: this pair's level-weight cotangent, ``(1, levels_width)``; entries
        at or beyond ``num_levels`` are padding and exactly zero.
    scalar_bar_ref : KernelRef
        Output: this pair's ``(softening_sq, G)`` cotangents in lanes 0 and 1 of a
        ``(1, VEC)`` row; lanes 2 and 3 are inert.
    num_levels : int
        Number of interaction levels. ``<= 0`` disables level weighting entirely.
    levels_width : int
        Padded width of the level-cotangent row (a power of two for the Triton
        lowering, ``>= max(num_levels, 1)``).
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.

    Returns
    -------
    None
        Writes through the six ``*_bar_ref`` outputs.
    """
    if num_levels <= 0:
        weight = None
        level_hits = None
    else:
        # The one-hot ladder is built once at the padded width so the level
        # cotangent row is written whole; levels >= num_levels are never hit.
        level_hits = _pair_level_hits(ra_ref[0], rb_ref[0], levels_width)
        table = jnp.reshape(lw_ref[...], (num_levels, 1, 1))
        weight = jnp.sum(table * level_hits[:num_levels].astype(table.dtype), axis=0)
    fa_bar_xyz = (fa_bar_ref[0, :, 0], fa_bar_ref[0, :, 1], fa_bar_ref[0, :, 2])
    if emit_b:
        fb_bar_xyz = (fb_bar_ref[0, :, 0], fb_bar_ref[0, :, 1], fb_bar_ref[0, :, 2])
    else:
        # The forward wrote zeros to the b side, so it carries no cotangent.
        zeros = jnp.zeros_like(fa_bar_xyz[0])
        fb_bar_xyz = (zeros, zeros, zeros)
    xa_bar, ma_bar, xb_bar, mb_bar, lw_bar, soft_bar, g_bar = _block_vjp_tiles(
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
        level_hits=level_hits,
    )
    zero = jnp.zeros_like(ma_bar)
    for lane in range(3):
        xa_bar_ref[0, :, lane] = xa_bar[lane]
        xb_bar_ref[0, :, lane] = xb_bar[lane]
    xa_bar_ref[0, :, 3] = zero
    xb_bar_ref[0, :, 3] = zero
    ma_bar_ref[0, :] = ma_bar
    mb_bar_ref[0, :] = mb_bar
    # Whole-row stores: a scalar store into one lane would be a masked store the
    # Triton lowering handles worse than a select over an iota.
    if lw_bar is None:
        lw_bar_ref[0, :] = jnp.zeros((levels_width,), dtype=ma_bar.dtype)
    else:
        lw_bar_ref[0, :] = lw_bar
    lane_index = lax.broadcasted_iota(jnp.int32, (_VEC_WIDTH,), 0)
    scalar_row = jnp.where(
        lane_index == 0,
        soft_bar,
        jnp.where(lane_index == 1, g_bar, jnp.zeros_like(soft_bar)),
    )
    scalar_bar_ref[0, :] = scalar_row.astype(ma_bar.dtype)


# FIRST ENFORCED ANNOTATIONS IN `jaccpot/pallas/`, AND THE BOUNDARY IS DELIBERATE.
#
# `_compat.py` records that nothing in this package carried `@jaxtyped`/`beartype`:
# its ~68 `KernelRef` annotations are documentation, present only because pydoclint
# will not let a parameter be *documented* until it is annotated. These five
# functions cross that line; the three tile helpers below do NOT, and the split is
# on one question -- does the check run inside a `pallas_call` body?
#
# It does not here. All five are ordinary traced Python: they run before
# `pallas_call`, take arrays rather than `KernelRef`s, and are what
# `mutual/nearfield.py` and `test_custom_vjp_parity.py` actually call. The check is
# a trace-time check, so it costs one comparison per compilation and nothing per
# step.
#
# COVERAGE BOUND, per STYLE_GUIDE section 4.3. `bench/annotation_pilot` measured
# these on CPU, where every recorded call carries `interpret=True` --
# `pallas_nearfield_mutual_supported()` requires an Ampere+ card. The shapes are
# fixed by the BlockSpecs and not by the backend, so the compiled Triton lane sees
# the same ones; that is an argument, not a measurement, and there is no GPU leg in
# CI to turn it into one. `python -m bench.gpu_gate` is where it gets checked.


@jaxtyped(typechecker=beartype)
def _pad_inputs(
    xa: Float[Array, "pairs w 3"],
    ma: Float[Array, "pairs w"],
    va_f: Float[Array, "pairs w"],
    rung_a_f: Optional[Float[Array, "pairs w"]],
    width: int,
    dtype: Any,
) -> tuple[Array, Array, Array, Array]:
    """Pad a leaf block's slot axis out to the Triton tile width.

    Padded slots get mask 0, so they are inert in every tile the kernels build.

    Parameters
    ----------
    xa : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` positions for one side of the pair list.
    ma : Float[Array, 'pairs w']
        ``(pairs, slots)`` masses.
    va_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` validity mask, float-encoded.
    rung_a_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` rungs, or ``None`` to substitute zeros.
    width : int
        Padded slot width; a power of two from :func:`_next_pow2`.
    dtype : Any
        Working dtype every returned array is cast to.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(positions, masses, validity, rungs)``. Positions are also padded on
        the component axis out to the vector width.
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


@jaxtyped(typechecker=beartype)
def mutual_leafpair_block_jax(
    xa: Float[Array, "pairs w 3"],
    ma: Float[Array, "pairs w"],
    va_f: Float[Array, "pairs w"],
    xb: Float[Array, "pairs w 3"],
    mb: Float[Array, "pairs w"],
    vb_f: Float[Array, "pairs w"],
    rung_a_f: Optional[Float[Array, "pairs w"]],
    rung_b_f: Optional[Float[Array, "pairs w"]],
    level_weights: Optional[Float[Array, "levels"]],
    softening_sq: Array,
    g_value: Array,
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
) -> tuple[Array, Array]:
    """Pure-jnp twin of the kernel: the correctness and AD oracle.

    Returns ``(F_a, F_b)`` of shape ``(pairs, slots, 3)``.

    Parameters
    ----------
    xa : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-A positions.
    ma : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A masses.
    va_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-B positions.
    mb : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B masses.
    vb_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B validity mask, same encoding.
    rung_a_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-A rung, float-encoded, or ``None``.
    rung_b_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-B rung, float-encoded, or ``None``.
    level_weights : Optional[Float[Array, 'levels']]
        ``(num_levels,)`` weight per interaction level. ``None`` -- like a
        ``None`` rung -- disables level weighting, giving every pair weight one.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.

    Returns
    -------
    tuple[Array, Array]
        ``(F_a, F_b)``, each ``(pairs, slots, 3)``.
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


@jaxtyped(typechecker=beartype)
def mutual_leafpair_block_pallas(
    xa: Float[Array, "pairs w 3"],
    ma: Float[Array, "pairs w"],
    va_f: Float[Array, "pairs w"],
    xb: Float[Array, "pairs w 3"],
    mb: Float[Array, "pairs w"],
    vb_f: Float[Array, "pairs w"],
    rung_a_f: Optional[Float[Array, "pairs w"]],
    rung_b_f: Optional[Float[Array, "pairs w"]],
    level_weights: Optional[Float[Array, "levels"]],
    softening_sq: Array,
    g_value: Array,
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
    interpret: bool = False,
    backend: str = "triton",
) -> tuple[Array, Array]:
    """One Pallas program per leaf pair; same semantics as the jnp twin.
    Not differentiable on its own -- ``pallas_call`` has no JVP or transpose
    rule. Use :func:`mutual_leafpair_block_cvjp`, which wires this forward to
    the analytic reverse below.

    Parameters
    ----------
    xa : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-A positions.
    ma : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A masses.
    va_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-B positions.
    mb : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B masses.
    vb_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B validity mask, same encoding.
    rung_a_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-A rung, float-encoded, or ``None``.
    rung_b_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-B rung, float-encoded, or ``None``.
    level_weights : Optional[Float[Array, 'levels']]
        ``(num_levels,)`` weight per interaction level. ``None`` -- like a
        ``None`` rung -- disables level weighting, giving every pair weight one.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.
    interpret : bool
        Run the kernel in interpret mode, which executes the real kernel logic
        on CPU and is how the parity tests reach it without a GPU.
    backend : str
        Pallas lowering backend.

    Returns
    -------
    tuple[Array, Array]
        ``(F_a, F_b)``, each ``(pairs, slots, 3)``, matching the jnp twin.
    """
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


@jaxtyped(typechecker=beartype)
def mutual_leafpair_block_vjp_pallas(
    xa: Float[Array, "pairs w 3"],
    ma: Float[Array, "pairs w"],
    va_f: Float[Array, "pairs w"],
    xb: Float[Array, "pairs w 3"],
    mb: Float[Array, "pairs w"],
    vb_f: Float[Array, "pairs w"],
    rung_a_f: Optional[Float[Array, "pairs w"]],
    rung_b_f: Optional[Float[Array, "pairs w"]],
    level_weights: Optional[Float[Array, "levels"]],
    softening_sq: Array,
    g_value: Array,
    fa_bar: Float[Array, "pairs w 3"],
    fb_bar: Float[Array, "pairs w 3"],
    *,
    exclude_diagonal: bool = False,
    emit_b: bool = True,
    interpret: bool = False,
    backend: str = "triton",
) -> tuple[Array, Array, Array, Array, Optional[Array], Array, Array]:
    """Hand-written analytic reverse, one Pallas program per leaf pair.

    Returns ``(xa_bar, ma_bar, xb_bar, mb_bar, lw_bar, soft_bar, g_bar)``. Never
    itself differentiated, so its intermediates stay tile-bounded: the ``S x S``
    tiles live in registers and the only HBM traffic is ``O(pairs * S)``, against
    the ``O(pairs * S^2)`` transient that linearizing the pure-JAX twin
    materialises. The three parameter cotangents come out of the kernel as one
    ``O(pairs * K)`` row per pair and are summed over pairs here.

    Parameters
    ----------
    xa : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-A positions.
    ma : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A masses.
    va_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-B positions.
    mb : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B masses.
    vb_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B validity mask, same encoding.
    rung_a_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-A rung, float-encoded, or ``None``.
    rung_b_f : Optional[Float[Array, 'pairs w']]
        ``(pairs, slots)`` leaf-B rung, float-encoded, or ``None``.
    level_weights : Optional[Float[Array, 'levels']]
        ``(num_levels,)`` weight per interaction level. ``None`` -- like a
        ``None`` rung -- disables level weighting, giving every pair weight one.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    fa_bar : Float[Array, 'pairs w 3']
        Cotangent of ``F_a``, ``(pairs, slots, 3)``.
    fb_bar : Float[Array, 'pairs w 3']
        Cotangent of ``F_b``, same shape.
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.
    interpret : bool
        Run the kernel in interpret mode.
    backend : str
        Pallas lowering backend.

    Returns
    -------
    tuple[Array, Array, Array, Array, Optional[Array], Array, Array]
        ``(xa_bar, ma_bar, xb_bar, mb_bar, lw_bar, soft_bar, g_bar)``: the block
        cotangents as before, the ``(num_levels,)`` level-weight cotangent (or
        ``None`` when ``level_weights`` is ``None``), and the scalar cotangents
        of ``softening_sq`` and ``g_value``, all summed over the pairs.
    """
    pairs, slots = int(ma.shape[0]), int(ma.shape[1])
    dtype = jnp.asarray(xa).dtype
    num_levels = 0 if level_weights is None else int(level_weights.shape[0])
    if pairs == 0:
        return (
            jnp.zeros((pairs, slots, 3), dtype=dtype),
            jnp.zeros((pairs, slots), dtype=dtype),
            jnp.zeros((pairs, slots, 3), dtype=dtype),
            jnp.zeros((pairs, slots), dtype=dtype),
            None if num_levels <= 0 else jnp.zeros((num_levels,), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
        )

    width = _next_pow2(slots)
    levels_width = _next_pow2(max(num_levels, 1))
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
        levels_width=levels_width,
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

    xa_bar, ma_bar, xb_bar, mb_bar, lw_bar_rows, scalar_rows = pl.pallas_call(
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
        out_specs=[
            bs_mat(),
            bs_vec(width),
            bs_mat(),
            bs_vec(width),
            bs_vec(levels_width),
            bs_vec(_VEC_WIDTH),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
            jax.ShapeDtypeStruct((pairs, width), dtype),
            jax.ShapeDtypeStruct((pairs, width, _VEC_WIDTH), dtype),
            jax.ShapeDtypeStruct((pairs, width), dtype),
            jax.ShapeDtypeStruct((pairs, levels_width), dtype),
            jax.ShapeDtypeStruct((pairs, _VEC_WIDTH), dtype),
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
    # Per-pair rows -> totals. Summed here, outside the kernel, because a
    # cross-program accumulation would need atomics the interpret path lacks.
    lw_bar = None if num_levels <= 0 else jnp.sum(lw_bar_rows[:, :num_levels], axis=0)
    scalar_bar = jnp.sum(scalar_rows, axis=0)
    return (
        xa_bar[:, :slots, :3],
        ma_bar[:, :slots],
        xb_bar[:, :slots, :3],
        mb_bar[:, :slots],
        lw_bar,
        scalar_bar[0],
        scalar_bar[1],
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
@jaxtyped(typechecker=beartype)
def mutual_leafpair_block_cvjp(
    xa: Float[Array, "pairs w 3"],
    ma: Float[Array, "pairs w"],
    va_f: Float[Array, "pairs w"],
    xb: Float[Array, "pairs w 3"],
    mb: Float[Array, "pairs w"],
    vb_f: Float[Array, "pairs w"],
    rung_a_f: Float[Array, "pairs w"],
    rung_b_f: Float[Array, "pairs w"],
    level_weights: Float[Array, "levels"],
    softening_sq: Array,
    g_value: Array,
    num_levels: int,
    exclude_diagonal: bool,
    emit_b: bool,
    interpret: bool,
) -> tuple[Array, Array]:
    """Differentiable mutual leaf-pair block: Pallas forward, analytic reverse.
    This is the entry point callers should use: it is the ``custom_vjp`` that
    pairs :func:`mutual_leafpair_block_pallas` with
    :func:`mutual_leafpair_block_vjp_pallas`. The validity masks and rungs are
    discrete and receive zero cotangents; ``level_weights``, ``softening_sq`` and
    ``g_value`` receive their analytic ones, which is what makes a loss
    differentiable with respect to the block-step ``dt_max`` (the weights are
    ``half * dt_max / 2**k``), the softening and ``G`` through this kernel.

    Parameters
    ----------
    xa : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-A positions.
    ma : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A masses.
    va_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A validity mask, float-encoded and tested ``> 0.5`` -- float rather than
        boolean because a Pallas ref carries the working dtype.
    xb : Float[Array, 'pairs w 3']
        ``(pairs, slots, 3)`` leaf-B positions.
    mb : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B masses.
    vb_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B validity mask, same encoding.
    rung_a_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-A rung, float-encoded.
    rung_b_f : Float[Array, 'pairs w']
        ``(pairs, slots)`` leaf-B rung, float-encoded.
    level_weights : Float[Array, 'levels']
        ``(num_levels,)`` weight per interaction level.
    softening_sq : Array
        Squared Plummer softening length, scalar.
    g_value : Array
        Gravitational constant, scalar.
    num_levels : int
        Number of interaction levels. ``<= 0`` disables level weighting entirely.
    exclude_diagonal : bool
        Drop ``i == j``, so a particle in a leaf paired with itself does not act
        on itself.
    emit_b : bool
        Also produce the ``b``-side output.
    interpret : bool
        Run the kernels in interpret mode.

    Returns
    -------
    tuple[Array, Array]
        ``(F_a, F_b)``, each ``(pairs, slots, 3)``.
    """
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
    xa_bar, ma_bar, xb_bar, mb_bar, lw_bar, soft_bar, g_bar = (
        mutual_leafpair_block_vjp_pallas(
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
    )
    # Masks and rungs are discrete: they take zero cotangents. They travel as
    # regular FLOAT array args -- never `nondiff_argnums`, which is for hashable
    # statics, and never as bool/int arrays, whose tangent type is `float0` and
    # cannot take a `zeros_like`. The level table is NOT discrete -- its entries
    # are `half * dt_max / 2**k`, smooth in dt_max, and the force is linear in
    # them -- so it, softening_sq and g_value get the analytic cotangents the
    # reverse kernel now emits. With `num_levels <= 0` the table is a one-entry
    # placeholder the forward never read, and its cotangent is zero.
    return (
        xa_bar,
        ma_bar,
        jnp.zeros_like(va_f),
        xb_bar,
        mb_bar,
        jnp.zeros_like(vb_f),
        jnp.zeros_like(rung_a_f),
        jnp.zeros_like(rung_b_f),
        (
            jnp.zeros_like(level_weights)
            if lw_bar is None
            else jnp.asarray(lw_bar, dtype=level_weights.dtype).reshape(
                level_weights.shape
            )
        ),
        jnp.asarray(soft_bar, dtype=jnp.asarray(softening_sq).dtype).reshape(
            jnp.shape(softening_sq)
        ),
        jnp.asarray(g_bar, dtype=jnp.asarray(g_value).dtype).reshape(
            jnp.shape(g_value)
        ),
    )


mutual_leafpair_block_cvjp.defvjp(
    _mutual_leafpair_block_cvjp_fwd, _mutual_leafpair_block_cvjp_bwd
)
