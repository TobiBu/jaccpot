"""Fused leaf-major near-field P2P kernel.

The pure-JAX radix fast lane evaluates cross-leaf particle-particle
contributions by materializing, per source-slot tile, a dense ``W_t x W_s``
distance matrix that XLA writes to HBM.  On GPU this is memory-bound: profiling
shows ~2,500 tiny (~1-2 us) kernels per step at ~7% of FLOP peak.

This module provides a fused alternative.  One Pallas program handles a single
*target leaf* (a vector of ``W_t`` target particles) and streams that leaf's
flattened source particles in a ``fori_loop``, accumulating acceleration and
potential in registers.  The ``W_t x W_s`` products live in registers/SRAM and
never touch HBM.  Output is leaf-major ``(num_leaves, W_t, 4)`` (acceleration in
lanes ``0:3``, potential in lane ``3``) so the caller can reuse the existing
scatter helpers unchanged.

The kernel computes cross-leaf pairs only; the intra-leaf ``i == j`` self term
stays on its separate path in ``near_field.py`` (source slots never contain the
target's own leaf), so no in-kernel diagonal exclusion is needed.  Invalid /
padded sources and target lanes contribute exactly ``0`` via masking, matching
the pure-JAX reference ``_pair_contributions_batched``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None


# Acceleration (3) + potential (1) packed into a width-4 output lane.
_OUT_WIDTH = 4
# Default target-subtile (vector width per program). 32 gives the best A100
# occupancy in benchmarks; the effective value is clamped to a power of two
# not exceeding the leaf width (Triton requires power-of-two array sizes).
_DEFAULT_TARGET_SUBTILE = 32


def _pow2_floor(x: int) -> int:
    """Largest power of two <= x (>= 1)."""
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _resolve_subtile(target_subtile, leaf_width: int) -> int:
    """Resolve the target-subtile vector width to a power of two <= leaf_width."""
    bt = int(target_subtile) if target_subtile else _DEFAULT_TARGET_SUBTILE
    bt = max(1, min(bt, int(leaf_width)))
    return _pow2_floor(bt)


# Positions are padded to width 4 for aligned vector loads.
_POS_WIDTH = 4


def pallas_nearfield_fused_supported() -> bool:
    """Return whether the active accelerator can run the fused leaf kernel."""

    if pl is None or plgpu is None:
        return False
    if jax.default_backend() != "gpu":
        return False
    try:
        device = jax.devices()[0]
    except Exception:  # pragma: no cover - backend discovery is environment-dependent
        return False

    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is None:
        return False
    return float(compute_capability) >= 8.0


@jax.jit
def nearfield_fused_leaf_jax(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Reference leaf-major fused near-field update in pure JAX.

    Parameters
    ----------
    target_positions:
        ``(num_leaves, W_t, 3)`` leaf-major target positions.
    target_mask:
        ``(num_leaves, W_t)`` boolean validity of each target lane.
    source_positions:
        ``(num_leaves, K, 3)`` flattened source positions for each target leaf
        (``K = num_source_slots * W_s``).
    source_masses:
        ``(num_leaves, K)`` flattened source masses.
    source_mask:
        ``(num_leaves, K)`` boolean validity of each flattened source.

    Returns
    -------
    Array
        ``(num_leaves, W_t, 4)`` with acceleration in lanes ``0:3`` and
        potential in lane ``3``.
    """

    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = lax.rsqrt(safe_dist_sq)
    inv_r = jnp.where(pair_mask, inv_r, 0.0)
    inv_dist3 = inv_r * inv_r * inv_r

    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
    potentials = jnp.where(target_mask, potentials, 0.0)

    return jnp.concatenate([accels, potentials[..., None]], axis=-1)


def _nearfield_fused_leaf_kernel(
    target_positions_ref,  # (1, W_t, _POS_WIDTH)
    target_mask_ref,  # (1, W_t)
    source_positions_ref,  # (1, K, _POS_WIDTH)
    source_masses_ref,  # (1, K)
    source_mask_ref,  # (1, K)
    softening_sq_ref,  # (1,)
    g_ref,  # (1,)
    out_ref,  # (1, W_t, _OUT_WIDTH)
    *,
    num_sources: int,
):
    """Fused near-field update for one target leaf (vector of W_t targets)."""

    tvalid = target_mask_ref[0, :]  # (W_t,)
    tx = target_positions_ref[0, :, 0]
    ty = target_positions_ref[0, :, 1]
    tz = target_positions_ref[0, :, 2]
    soft = softening_sq_ref[0]
    g_value = g_ref[0]

    zero = jnp.zeros_like(tx)
    acc0 = (zero, zero, zero, zero)

    def _body(k, acc):
        acc_x, acc_y, acc_z, acc_p = acc
        svalid = source_mask_ref[0, k]  # scalar
        sx = source_positions_ref[0, k, 0]
        sy = source_positions_ref[0, k, 1]
        sz = source_positions_ref[0, k, 2]
        sm = source_masses_ref[0, k]

        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        dist_sq = dx * dx + dy * dy + dz * dz + soft
        active = tvalid & svalid  # (W_t,), broadcast scalar svalid
        safe_dist_sq = jnp.where(active, dist_sq, 1.0)
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(active, inv_r, 0.0)
        inv_dist3 = inv_r * inv_r * inv_r  # already 0 where inactive

        scale = -g_value * inv_dist3 * sm
        acc_x = acc_x + scale * dx
        acc_y = acc_y + scale * dy
        acc_z = acc_z + scale * dz
        acc_p = acc_p - g_value * inv_r * sm
        return (acc_x, acc_y, acc_z, acc_p)

    acc_x, acc_y, acc_z, acc_p = lax.fori_loop(0, num_sources, _body, acc0)

    out_ref[0, :, 0] = jnp.where(tvalid, acc_x, zero)
    out_ref[0, :, 1] = jnp.where(tvalid, acc_y, zero)
    out_ref[0, :, 2] = jnp.where(tvalid, acc_z, zero)
    out_ref[0, :, 3] = jnp.where(tvalid, acc_p, zero)


def nearfield_fused_leaf_pallas(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
    interpret: bool = False,
) -> Array:
    """Fused leaf-major near-field update with Pallas.

    See :func:`nearfield_fused_leaf_jax` for the argument/return contract.

    ``target_subtile`` splits each leaf's ``W_t`` targets into subtiles handled
    by separate programs, raising the grid size from ``num_leaves`` to
    ``num_leaves * ceil(W_t / target_subtile)``. This is the primary occupancy
    knob: with large leaves (e.g. ``W_t=256``) a per-leaf grid launches too few
    programs to fill the SMs. Sources are shared across a leaf's subtiles (same
    source block), so L2 reuse is preserved.
    """

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    target_positions = jnp.asarray(target_positions)
    dtype = target_positions.dtype
    source_positions = jnp.asarray(source_positions, dtype=dtype)
    target_mask = jnp.asarray(target_mask, dtype=bool)
    source_mask = jnp.asarray(source_mask, dtype=bool)
    source_masses = jnp.asarray(source_masses, dtype=dtype)
    softening_sq_arr = jnp.asarray([softening_sq], dtype=dtype)
    g_arr = jnp.asarray([G], dtype=dtype)

    if target_positions.ndim != 3 or target_positions.shape[-1] != 3:
        raise ValueError("target_positions must have shape (num_leaves, W_t, 3)")
    if source_positions.ndim != 3 or source_positions.shape[-1] != 3:
        raise ValueError("source_positions must have shape (num_leaves, K, 3)")

    num_leaves = int(target_positions.shape[0])
    tile_t = int(target_positions.shape[1])
    num_sources = int(source_positions.shape[1])

    if num_leaves == 0 or tile_t == 0 or num_sources == 0:
        return jnp.zeros((num_leaves, tile_t, _OUT_WIDTH), dtype=dtype)

    # Target-subtile size (targets handled per program). Pad W_t up to a
    # multiple of the subtile so the grid tiles evenly.
    bt = _resolve_subtile(target_subtile, tile_t)
    tile_t_pad = ((tile_t + bt - 1) // bt) * bt
    n_sub = tile_t_pad // bt
    pad_t = tile_t_pad - tile_t

    target_positions_padded = jnp.pad(
        target_positions, ((0, 0), (0, pad_t), (0, _POS_WIDTH - 3))
    )
    target_mask_padded = jnp.pad(target_mask, ((0, 0), (0, pad_t)))
    source_positions_padded = jnp.pad(
        source_positions, ((0, 0), (0, 0), (0, _POS_WIDTH - 3))
    )

    if num_warps is None:
        # One warp (32 threads) per 32 target lanes in a subtile; keep >= 1.
        num_warps = max(1, bt // 32)

    def _kernel(*refs):
        return _nearfield_fused_leaf_kernel(*refs, num_sources=num_sources)

    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((num_leaves, tile_t_pad, _OUT_WIDTH), dtype),
        in_specs=[
            pl.BlockSpec((1, bt, _POS_WIDTH), lambda leaf, sub: (leaf, sub, 0)),
            pl.BlockSpec((1, bt), lambda leaf, sub: (leaf, sub)),
            pl.BlockSpec((1, num_sources, _POS_WIDTH), lambda leaf, sub: (leaf, 0, 0)),
            pl.BlockSpec((1, num_sources), lambda leaf, sub: (leaf, 0)),
            pl.BlockSpec((1, num_sources), lambda leaf, sub: (leaf, 0)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
        ],
        out_specs=pl.BlockSpec((1, bt, _OUT_WIDTH), lambda leaf, sub: (leaf, sub, 0)),
        grid=(num_leaves, n_sub),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=f"nearfield_fused_leaf_t{bt}_k{num_sources}",
    )
    out = kernel(
        target_positions_padded,
        target_mask_padded,
        source_positions_padded,
        source_masses,
        source_mask,
        softening_sq_arr,
        g_arr,
    )
    if pad_t:
        out = out[:, :tile_t, :]
    return out


def nearfield_fused_leaf(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
    prefer_pallas: bool = True,
    interpret: bool = False,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
) -> Array:
    """Fused leaf-major near-field update using the best available backend."""

    use_pallas = interpret or (prefer_pallas and pallas_nearfield_fused_supported())
    if use_pallas and pl is not None:
        return nearfield_fused_leaf_pallas(
            target_positions,
            target_mask,
            source_positions,
            source_masses,
            source_mask,
            softening_sq=softening_sq,
            G=G,
            num_warps=num_warps,
            num_stages=num_stages,
            target_subtile=target_subtile,
            interpret=interpret,
        )
    return nearfield_fused_leaf_jax(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        softening_sq=softening_sq,
        G=G,
    )


def nearfield_fused_leaf_backend(*, prefer_pallas: bool = True) -> str:
    """Describe which backend :func:`nearfield_fused_leaf` will use."""

    if prefer_pallas and pallas_nearfield_fused_supported():
        return "pallas"
    return "jax"


# ---------------------------------------------------------------------------
# Leaf-pair kernel: consumes the compact source-*leaf-id* layout (the
# production fused near-field path). Instead of materializing a dense
# (num_leaves, num_source_slots, W_s) source-particle tensor -- which is ~99%
# padding and OOMs at large leaf sizes -- this kernel keeps sources as leaf ids
# and gathers each source leaf's W particles from ``leaf_positions`` (~N*3*4
# bytes total) inside the kernel, skipping invalid source slots via lax.cond.
# ---------------------------------------------------------------------------


@jax.jit
def nearfield_leafpair_jax(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    source_leaf_ids: Array,
    source_valid: Array,
    *,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Reference leaf-pair near-field update in pure JAX (dense; test-scale only).

    Parameters
    ----------
    leaf_positions:
        ``(num_leaves, W, 3)`` leaf-major particle positions (target = source).
    leaf_masses / leaf_mask:
        ``(num_leaves, W)`` per-particle mass / validity.
    source_leaf_ids:
        ``(num_leaves, S)`` neighbour source-leaf ids for each target leaf.
    source_valid:
        ``(num_leaves, S)`` validity of each source slot.

    Returns
    -------
    Array
        ``(num_leaves, W, 4)`` leaf-major acceleration (0:3) + potential (3).
    """
    safe_sids = jnp.where(source_valid, source_leaf_ids, 0)
    src_pos = leaf_positions[safe_sids]  # (L, S, W, 3)
    src_mass = leaf_masses[safe_sids]  # (L, S, W)
    src_valid = leaf_mask[safe_sids] & source_valid[:, :, None]  # (L, S, W)

    # target (L, W_t, 1, 1, 3) vs source (L, 1, S, W_s, 3)
    diff = leaf_positions[:, :, None, None, :] - src_pos[:, None, :, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq  # (L, W_t, S, W_s)
    pair_mask = leaf_mask[:, :, None, None] & src_valid[:, None, :, :]
    safe_dist_sq = jnp.where(pair_mask, dist_sq, 1.0)
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    inv_dist3 = inv_r * inv_r * inv_r
    weighted = inv_dist3 * src_mass[:, None, :, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=(2, 3))  # (L, W_t, 3)
    accels = jnp.where(leaf_mask[..., None], accels, 0.0)
    potentials = -G * jnp.sum(inv_r * src_mass[:, None, :, :], axis=(2, 3))
    potentials = jnp.where(leaf_mask, potentials, 0.0)
    return jnp.concatenate([accels, potentials[..., None]], axis=-1)


def _nearfield_leafpair_kernel(
    target_positions_ref,  # (1, Bt, _POS_WIDTH)
    target_mask_ref,  # (1, Bt)
    src_table_pos_ref,  # (L, W, _POS_WIDTH) full gather table
    src_table_mass_ref,  # (L, W)
    src_table_mask_ref,  # (L, W)
    source_leaf_ids_ref,  # (1, S)
    source_valid_ref,  # (1, S)
    softening_sq_ref,  # (1,)
    g_ref,  # (1,)
    out_ref,  # (1, Bt, _OUT_WIDTH)
    *,
    num_source_slots: int,
    leaf_width: int,
):
    """Leaf-pair near-field update for one target subtile (vector of Bt targets)."""

    tvalid = target_mask_ref[0, :]
    tx = target_positions_ref[0, :, 0]
    ty = target_positions_ref[0, :, 1]
    tz = target_positions_ref[0, :, 2]
    soft = softening_sq_ref[0]
    g_value = g_ref[0]

    zero = jnp.zeros_like(tx)
    acc0 = (zero, zero, zero, zero)

    def _slot_body(s, acc):
        sid = source_leaf_ids_ref[0, s]
        slot_valid = source_valid_ref[0, s]

        def _apply(acc):
            def _lane_body(j, acc):
                acc_x, acc_y, acc_z, acc_p = acc
                sx = src_table_pos_ref[sid, j, 0]
                sy = src_table_pos_ref[sid, j, 1]
                sz = src_table_pos_ref[sid, j, 2]
                sm = src_table_mass_ref[sid, j]
                lane_valid = src_table_mask_ref[sid, j]
                dx = tx - sx
                dy = ty - sy
                dz = tz - sz
                dist_sq = dx * dx + dy * dy + dz * dz + soft
                active = tvalid & lane_valid
                safe_dist_sq = jnp.where(active, dist_sq, 1.0)
                inv_r = lax.rsqrt(safe_dist_sq)
                inv_r = jnp.where(active, inv_r, 0.0)
                inv_dist3 = inv_r * inv_r * inv_r
                scale = -g_value * inv_dist3 * sm
                acc_x = acc_x + scale * dx
                acc_y = acc_y + scale * dy
                acc_z = acc_z + scale * dz
                acc_p = acc_p - g_value * inv_r * sm
                return (acc_x, acc_y, acc_z, acc_p)

            return lax.fori_loop(0, leaf_width, _lane_body, acc)

        return lax.cond(slot_valid, _apply, lambda acc: acc, acc)

    acc_x, acc_y, acc_z, acc_p = lax.fori_loop(0, num_source_slots, _slot_body, acc0)

    out_ref[0, :, 0] = jnp.where(tvalid, acc_x, zero)
    out_ref[0, :, 1] = jnp.where(tvalid, acc_y, zero)
    out_ref[0, :, 2] = jnp.where(tvalid, acc_z, zero)
    out_ref[0, :, 3] = jnp.where(tvalid, acc_p, zero)


def nearfield_leafpair_pallas(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    source_leaf_ids: Array,
    source_valid: Array,
    *,
    softening_sq: Array,
    G: Array,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
    interpret: bool = False,
) -> Array:
    """Leaf-pair near-field update with Pallas.

    See :func:`nearfield_leafpair_jax` for the argument/return contract. Source
    leaves are gathered by id from ``leaf_positions`` inside the kernel; invalid
    source slots are skipped with ``lax.cond`` so heavily-padded slot tensors
    cost only a cheap per-slot predicate check.
    """

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    leaf_positions = jnp.asarray(leaf_positions)
    dtype = leaf_positions.dtype
    leaf_masses = jnp.asarray(leaf_masses, dtype=dtype)
    leaf_mask = jnp.asarray(leaf_mask, dtype=bool)
    source_leaf_ids = jnp.asarray(source_leaf_ids)
    source_valid = jnp.asarray(source_valid, dtype=bool)
    softening_sq_arr = jnp.asarray([softening_sq], dtype=dtype)
    g_arr = jnp.asarray([G], dtype=dtype)

    if leaf_positions.ndim != 3 or leaf_positions.shape[-1] != 3:
        raise ValueError("leaf_positions must have shape (num_leaves, W, 3)")

    num_leaves = int(leaf_positions.shape[0])
    leaf_width = int(leaf_positions.shape[1])
    num_source_slots = int(source_leaf_ids.shape[1])

    if num_leaves == 0 or leaf_width == 0 or num_source_slots == 0:
        return jnp.zeros((num_leaves, leaf_width, _OUT_WIDTH), dtype=dtype)

    leaf_positions_padded = jnp.pad(
        leaf_positions, ((0, 0), (0, 0), (0, _POS_WIDTH - 3))
    )

    bt = _resolve_subtile(target_subtile, leaf_width)
    width_pad = ((leaf_width + bt - 1) // bt) * bt
    n_sub = width_pad // bt
    pad_t = width_pad - leaf_width

    target_positions_padded = (
        jnp.pad(leaf_positions_padded, ((0, 0), (0, pad_t), (0, 0)))
        if pad_t
        else leaf_positions_padded
    )
    target_mask_padded = jnp.pad(leaf_mask, ((0, 0), (0, pad_t)))

    if num_warps is None:
        num_warps = max(1, bt // 32)

    def _kernel(*refs):
        return _nearfield_leafpair_kernel(
            *refs, num_source_slots=num_source_slots, leaf_width=leaf_width
        )

    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((num_leaves, width_pad, _OUT_WIDTH), dtype),
        in_specs=[
            pl.BlockSpec((1, bt, _POS_WIDTH), lambda leaf, sub: (leaf, sub, 0)),
            pl.BlockSpec((1, bt), lambda leaf, sub: (leaf, sub)),
            # Full gather tables (indexed by data-dependent source leaf id).
            pl.BlockSpec(
                (num_leaves, leaf_width, _POS_WIDTH), lambda leaf, sub: (0, 0, 0)
            ),
            pl.BlockSpec((num_leaves, leaf_width), lambda leaf, sub: (0, 0)),
            pl.BlockSpec((num_leaves, leaf_width), lambda leaf, sub: (0, 0)),
            pl.BlockSpec((1, num_source_slots), lambda leaf, sub: (leaf, 0)),
            pl.BlockSpec((1, num_source_slots), lambda leaf, sub: (leaf, 0)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
            pl.BlockSpec((1,), lambda leaf, sub: (0,)),
        ],
        out_specs=pl.BlockSpec((1, bt, _OUT_WIDTH), lambda leaf, sub: (leaf, sub, 0)),
        grid=(num_leaves, n_sub),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=f"nearfield_leafpair_t{bt}_s{num_source_slots}_w{leaf_width}",
    )
    out = kernel(
        target_positions_padded,
        target_mask_padded,
        leaf_positions_padded,
        leaf_masses,
        leaf_mask,
        source_leaf_ids,
        source_valid,
        softening_sq_arr,
        g_arr,
    )
    if pad_t:
        out = out[:, :leaf_width, :]
    return out


__all__ = [
    "nearfield_fused_leaf",
    "nearfield_fused_leaf_backend",
    "nearfield_fused_leaf_jax",
    "nearfield_fused_leaf_pallas",
    "nearfield_leafpair_jax",
    "nearfield_leafpair_pallas",
    "pallas_nearfield_fused_supported",
]
