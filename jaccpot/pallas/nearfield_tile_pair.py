"""Fused nearfield target-tile x source-tile kernels."""

from __future__ import annotations

import math

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


_PADDED_VECTOR_WIDTH = 4


def pallas_nearfield_tile_pair_supported() -> bool:
    """Return whether the active accelerator can run the fused tile kernel."""

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
def nearfield_tile_pair_accel_jax(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Reference fused tile-pair acceleration update in pure JAX."""

    diff = target_positions[:, None, :] - source_positions[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, None] & source_mask[None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = lax.rsqrt(safe_dist_sq)
    inv_r = jnp.where(pair_mask, inv_r, 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)

    weighted = inv_dist3 * source_masses[None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=1)
    return jnp.where(target_mask[:, None], accels, 0.0)


def _nearfield_tile_pair_kernel(
    target_positions_ref,
    target_mask_ref,
    source_positions_ref,
    source_masses_ref,
    source_mask_ref,
    softening_sq_ref,
    g_ref,
    out_ref,
):
    """Compute one fused target-tile x source-tile interaction block."""

    target_idx = pl.program_id(axis=0)
    target_valid = plgpu.load(
        target_mask_ref.at[target_idx],
        mask=jnp.array(True),
        other=False,
    )
    target_x = plgpu.load(
        target_positions_ref.at[target_idx, 0],
        mask=jnp.array(True),
        other=0.0,
    )
    target_y = plgpu.load(
        target_positions_ref.at[target_idx, 1],
        mask=jnp.array(True),
        other=0.0,
    )
    target_z = plgpu.load(
        target_positions_ref.at[target_idx, 2],
        mask=jnp.array(True),
        other=0.0,
    )
    softening_sq = plgpu.load(
        softening_sq_ref.at[0],
        mask=jnp.array(True),
        other=0.0,
    )
    g_value = plgpu.load(
        g_ref.at[0],
        mask=jnp.array(True),
        other=0.0,
    )
    zero = jnp.asarray(0.0, dtype=target_positions_ref.dtype)
    acc0 = (zero, zero, zero)

    def _body(src_idx: int, acc: Array) -> Array:
        acc_x, acc_y, acc_z = acc
        src_valid = plgpu.load(
            source_mask_ref.at[src_idx],
            mask=jnp.array(True),
            other=False,
        )
        src_x = plgpu.load(
            source_positions_ref.at[src_idx, 0],
            mask=jnp.array(True),
            other=0.0,
        )
        src_y = plgpu.load(
            source_positions_ref.at[src_idx, 1],
            mask=jnp.array(True),
            other=0.0,
        )
        src_z = plgpu.load(
            source_positions_ref.at[src_idx, 2],
            mask=jnp.array(True),
            other=0.0,
        )
        src_mass = plgpu.load(
            source_masses_ref.at[src_idx],
            mask=jnp.array(True),
            other=0.0,
        )
        diff_x = target_x - src_x
        diff_y = target_y - src_y
        diff_z = target_z - src_z
        dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + softening_sq
        active = target_valid & src_valid
        safe_dist_sq = jnp.where(active, dist_sq, 1.0)
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(active, inv_r, 0.0)
        inv_dist3 = jnp.where(active, inv_r * inv_r * inv_r, 0.0)
        scale = -g_value * inv_dist3 * src_mass
        next_x = jnp.where(active, acc_x + scale * diff_x, acc_x)
        next_y = jnp.where(active, acc_y + scale * diff_y, acc_y)
        next_z = jnp.where(active, acc_z + scale * diff_z, acc_z)
        return (
            next_x,
            next_y,
            next_z,
        )

    acc_x, acc_y, acc_z = lax.fori_loop(0, source_positions_ref.shape[0], _body, acc0)
    plgpu.store(
        out_ref.at[target_idx, 0],
        jnp.where(target_valid, acc_x, zero),
        mask=jnp.array(True),
    )
    plgpu.store(
        out_ref.at[target_idx, 1],
        jnp.where(target_valid, acc_y, zero),
        mask=jnp.array(True),
    )
    plgpu.store(
        out_ref.at[target_idx, 2],
        jnp.where(target_valid, acc_z, zero),
        mask=jnp.array(True),
    )
    plgpu.store(
        out_ref.at[target_idx, 3],
        zero,
        mask=jnp.array(True),
    )


def nearfield_tile_pair_accel_pallas(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Compute one fused target-tile x source-tile block with Pallas."""

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    target_positions = jnp.asarray(target_positions)
    source_positions = jnp.asarray(source_positions)
    target_mask = jnp.asarray(target_mask, dtype=bool)
    source_mask = jnp.asarray(source_mask, dtype=bool)
    source_masses = jnp.asarray(source_masses, dtype=target_positions.dtype)
    softening_sq_arr = jnp.asarray([softening_sq], dtype=target_positions.dtype)
    g_arr = jnp.asarray([G], dtype=target_positions.dtype)

    if target_positions.ndim != 2 or target_positions.shape[-1] != 3:
        raise ValueError("target_positions must have shape (tile, 3)")
    if source_positions.ndim != 2 or source_positions.shape[-1] != 3:
        raise ValueError("source_positions must have shape (tile, 3)")
    if target_positions.shape[0] != source_positions.shape[0]:
        raise ValueError("target and source tiles must have the same tile size")

    tile_size = int(target_positions.shape[0])
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(
            "Pallas tile kernel requires a positive power-of-two tile size"
        )

    target_positions_padded = jnp.pad(target_positions, ((0, 0), (0, 1)))
    source_positions_padded = jnp.pad(source_positions, ((0, 0), (0, 1)))

    kernel = pl.pallas_call(
        _nearfield_tile_pair_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (tile_size, _PADDED_VECTOR_WIDTH),
            target_positions.dtype,
        ),
        in_specs=[
            pl.BlockSpec((tile_size, _PADDED_VECTOR_WIDTH), lambda pid: (0, 0)),
            pl.BlockSpec((tile_size,), lambda pid: (0,)),
            pl.BlockSpec((tile_size, _PADDED_VECTOR_WIDTH), lambda pid: (0, 0)),
            pl.BlockSpec((tile_size,), lambda pid: (0,)),
            pl.BlockSpec((tile_size,), lambda pid: (0,)),
            pl.BlockSpec((1,), lambda pid: (0,)),
            pl.BlockSpec((1,), lambda pid: (0,)),
        ],
        out_specs=pl.BlockSpec((tile_size, _PADDED_VECTOR_WIDTH), lambda pid: (0, 0)),
        grid=(tile_size,),
        compiler_params=plgpu.CompilerParams(
            num_warps=max(1, tile_size // 8), num_stages=1
        ),
        interpret=False,
        name=f"nearfield_tile_pair_t{tile_size}",
    )
    return kernel(
        target_positions_padded,
        target_mask,
        source_positions_padded,
        source_masses,
        source_mask,
        softening_sq_arr,
        g_arr,
    )[:, :3]


def nearfield_tile_pair_accel(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Array,
    G: Array,
    prefer_pallas: bool = True,
) -> Array:
    """Compute one fused target/source tile pair with the best available backend."""

    if prefer_pallas and pallas_nearfield_tile_pair_supported():
        return nearfield_tile_pair_accel_pallas(
            target_positions,
            target_mask,
            source_positions,
            source_masses,
            source_mask,
            softening_sq=softening_sq,
            G=G,
        )
    return nearfield_tile_pair_accel_jax(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        softening_sq=softening_sq,
        G=G,
    )


def nearfield_tile_pair_backend(*, prefer_pallas: bool = True) -> str:
    """Describe which backend `nearfield_tile_pair_accel` will use."""

    if prefer_pallas and pallas_nearfield_tile_pair_supported():
        return "pallas"
    return "jax"


__all__ = [
    "nearfield_tile_pair_accel",
    "nearfield_tile_pair_accel_jax",
    "nearfield_tile_pair_accel_pallas",
    "nearfield_tile_pair_backend",
    "pallas_nearfield_tile_pair_supported",
]
