"""Packed unique-update helpers for nearfield particle accumulation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None


_PADDED_VECTOR_WIDTH = 4


def pallas_nearfield_unique_updates_supported() -> bool:
    """Return whether the active accelerator can run the Triton update kernel."""

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
def pack_unique_particle_vector_updates(
    indices: Array,
    values: Array,
    mask: Array,
) -> tuple[Array, Array, Array]:
    """Pack repeated particle updates into sorted unique rows.

    The output keeps a fixed padded shape for JIT stability: the leading
    dimension matches the flattened input size, with invalid trailing rows
    marked by `unique_valid=False`.
    """

    if values.size == 0:
        empty_index = jnp.zeros((0,), dtype=INDEX_DTYPE)
        empty_values = jnp.zeros((0, values.shape[-1]), dtype=values.dtype)
        empty_valid = jnp.zeros((0,), dtype=bool)
        return empty_index, empty_values, empty_valid

    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)

    encoded_indices = jnp.where(flat_mask, flat_indices + 1, 0).astype(INDEX_DTYPE)
    sort_idx = jnp.argsort(encoded_indices, stable=True)
    encoded_sorted = encoded_indices[sort_idx]
    values_sorted = jnp.where(flat_mask[:, None], flat_values, 0.0)[sort_idx]
    valid_sorted = encoded_sorted > 0

    item_count = encoded_sorted.shape[0]
    is_new = jnp.concatenate(
        [
            jnp.array([True], dtype=bool),
            encoded_sorted[1:] != encoded_sorted[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - 1

    unique_encoded = (
        jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(encoded_sorted)
    )
    unique_valid = jnp.zeros((item_count,), dtype=bool).at[group_ids].set(valid_sorted)
    unique_values = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    unique_indices = jnp.where(unique_valid, unique_encoded - 1, 0)
    return unique_indices, unique_values, unique_valid


@jax.jit
def apply_packed_particle_vector_updates_jax(
    base_acc: Array,
    unique_indices: Array,
    unique_values: Array,
    unique_valid: Array,
) -> Array:
    """Apply packed unique updates with gather-add-set instead of scatter-add."""

    if unique_values.size == 0:
        return base_acc

    safe_indices = jnp.where(unique_valid, unique_indices, 0)
    masked_values = jnp.where(unique_valid[:, None], unique_values, 0.0)
    gathered = base_acc[safe_indices]
    updated = gathered + masked_values
    safe_rows = jnp.where(unique_valid[:, None], updated, gathered)
    return base_acc.at[safe_indices].set(safe_rows)


def _nearfield_unique_updates_kernel(
    indices_ref,
    values_ref,
    base_ref,
    out_ref,
    *,
    block_size: int,
):
    """Store one packed unique-update block into the dense particle buffer."""

    update_idx = pl.program_id(axis=0) * block_size + jnp.arange(block_size)
    row_mask = update_idx < indices_ref.shape[0]
    component_idx = jnp.arange(_PADDED_VECTOR_WIDTH)

    particle_idx = plgpu.load(indices_ref.at[update_idx], mask=row_mask, other=0)
    values = plgpu.load(
        values_ref.at[update_idx[:, None], component_idx[None, :]],
        mask=row_mask[:, None],
        other=0.0,
    )
    base = plgpu.load(
        base_ref.at[particle_idx[:, None], component_idx[None, :]],
        mask=row_mask[:, None],
        other=0.0,
    )
    plgpu.store(
        out_ref.at[particle_idx[:, None], component_idx[None, :]],
        base + values,
        mask=row_mask[:, None],
    )


def _apply_packed_particle_vector_updates_pallas(
    base_acc: Array,
    unique_indices: Array,
    unique_values: Array,
    unique_valid: Array,
) -> Array:
    """Apply packed unique updates with a Triton-backed Pallas kernel."""

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    if base_acc.ndim != 2 or base_acc.shape[1] != 3:
        raise ValueError("base_acc must have shape (num_particles, 3)")

    safe_indices = jnp.where(unique_valid, unique_indices, 0)
    masked_values = jnp.where(unique_valid[:, None], unique_values, 0.0)
    padded_base = jnp.pad(base_acc, ((0, 0), (0, 1)))
    padded_values = jnp.pad(masked_values, ((0, 0), (0, 1)))
    padded_values = jnp.asarray(padded_values, dtype=base_acc.dtype)

    block_size = 128
    update_count = int(safe_indices.shape[0])
    particle_count = int(base_acc.shape[0])
    kernel = pl.pallas_call(
        _nearfield_unique_updates_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (particle_count, _PADDED_VECTOR_WIDTH), base_acc.dtype
        ),
        in_specs=[
            pl.BlockSpec((block_size,), lambda pid: (pid * block_size,)),
            pl.BlockSpec(
                (block_size, _PADDED_VECTOR_WIDTH),
                lambda pid: (pid * block_size, 0),
            ),
            pl.BlockSpec((particle_count, _PADDED_VECTOR_WIDTH), lambda pid: (0, 0)),
        ],
        out_specs=pl.BlockSpec(
            (particle_count, _PADDED_VECTOR_WIDTH), lambda pid: (0, 0)
        ),
        grid=(math.ceil(update_count / block_size),),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=False,
        name=f"nearfield_unique_updates_{particle_count}_{update_count}",
    )
    return kernel(safe_indices, padded_values, padded_base)[:, :3]


def apply_packed_particle_vector_updates(
    base_acc: Array,
    unique_indices: Array,
    unique_values: Array,
    unique_valid: Array,
    *,
    prefer_pallas: bool = True,
) -> Array:
    """Apply packed unique updates with the best available backend."""

    if prefer_pallas and pallas_nearfield_unique_updates_supported():
        return _apply_packed_particle_vector_updates_pallas(
            base_acc,
            unique_indices,
            unique_values,
            unique_valid,
        )
    return apply_packed_particle_vector_updates_jax(
        base_acc,
        unique_indices,
        unique_values,
        unique_valid,
    )


def nearfield_unique_updates_backend(*, prefer_pallas: bool = True) -> str:
    """Describe which backend `apply_packed_particle_vector_updates` will use."""

    if prefer_pallas and pallas_nearfield_unique_updates_supported():
        return "pallas"
    return "jax_set"


__all__ = [
    "apply_packed_particle_vector_updates",
    "apply_packed_particle_vector_updates_jax",
    "nearfield_unique_updates_backend",
    "pack_unique_particle_vector_updates",
    "pallas_nearfield_unique_updates_supported",
]
