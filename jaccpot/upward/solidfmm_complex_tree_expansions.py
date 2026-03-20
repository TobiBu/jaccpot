"""SolidFMM-style complex-harmonic tree expansion helpers.

This module implements the *upward sweep* (P2M + M2M) for the complex
solid-harmonic basis used by the solidfmm-style backend. It is intentionally
kept separate from the Dehnen real-basis implementation.
"""

from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from yggdrax.geometry import TreeGeometry, compute_tree_geometry
from yggdrax.tree import Tree, get_level_offsets, get_nodes_by_level
from yggdrax.tree_moments import TreeMassMoments, compute_tree_mass_moments

from jaccpot.operators.complex_harmonics import p2m_complex_batch
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry,
    enforce_conjugate_symmetry_batch,
    m2m_complex,
    regular_solid_harmonic_directional_derivative_order_batch,
)
from jaccpot.operators.real_harmonics import sh_size


class SolidFMMComplexNodeMultipoleData(NamedTuple):
    """Packed complex multipole coefficients and their metadata."""

    order: int
    centers: Array  # (num_nodes, 3)
    packed: Array  # (num_nodes, (p+1)^2)
    source_motion_packed: Optional[Array]  # (num_nodes, (p+1)^2) or None


class SolidFMMComplexTreeUpwardData(NamedTuple):
    """Container bundling data needed for the complex upward sweep."""

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: SolidFMMComplexNodeMultipoleData


_CENTER_MODES = ("com", "aabb", "explicit")
_DEFAULT_LEAF_BATCH_SIZE = 2048


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


_UPWARD_DIAGNOSTICS = _env_flag("JACCPOT_PREPARE_DIAGNOSTICS", False)


def _upward_diag(message: str) -> None:
    if _UPWARD_DIAGNOSTICS:
        print(f"[jaccpot.upward] {message}", flush=True)


def _format_bytes(count: int) -> str:
    value = float(max(int(count), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _diag_upward_stage_estimates(
    *,
    num_particles: int,
    total_nodes: int,
    num_leaves: int,
    max_leaf_size: int,
    leaf_batch_size: int,
    coeffs: int,
    positions_dtype: jnp.dtype,
    masses_dtype: jnp.dtype,
) -> None:
    if not _UPWARD_DIAGNOSTICS:
        return

    pos_itemsize = np.dtype(positions_dtype).itemsize
    mass_itemsize = np.dtype(masses_dtype).itemsize
    complex_itemsize = np.dtype(
        complex_dtype_for_real(jnp.result_type(positions_dtype, masses_dtype))
    ).itemsize

    mass_prefix_bytes = (num_particles + 1) * mass_itemsize
    weighted_prefix_bytes = (num_particles + 1) * 3 * pos_itemsize
    total_mass_bytes = total_nodes * mass_itemsize
    center_bytes = total_nodes * 3 * pos_itemsize
    effective_batch = max(1, min(int(leaf_batch_size), int(num_leaves)))
    leaf_point_bytes = effective_batch * max_leaf_size * 3 * pos_itemsize
    leaf_mass_bytes = effective_batch * max_leaf_size * mass_itemsize
    leaf_contrib_bytes = effective_batch * max_leaf_size * coeffs * complex_itemsize
    packed_bytes = total_nodes * coeffs * complex_itemsize

    _upward_diag(
        "stage estimates "
        f"mass_prefix={_format_bytes(mass_prefix_bytes)} "
        f"weighted_prefix={_format_bytes(weighted_prefix_bytes)} "
        f"total_mass={_format_bytes(total_mass_bytes)} "
        f"centers={_format_bytes(center_bytes)} "
        f"p2m_leaf_points={_format_bytes(leaf_point_bytes)} "
        f"p2m_leaf_masses={_format_bytes(leaf_mass_bytes)} "
        f"p2m_leaf_contribs={_format_bytes(leaf_contrib_bytes)} "
        f"packed={_format_bytes(packed_bytes)}"
    )


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "max_leaf_size",
        "num_internal",
        "total_nodes",
        "leaf_batch_size",
    ),
)
def _p2m_leaves_complex(
    node_ranges: Array,
    positions_sorted: Array,
    masses_sorted: Array,
    centers: Array,
    *,
    order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
    leaf_batch_size: int,
) -> Array:
    """Leaf P2M for solidfmm-style complex SH coefficients."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    dtype = complex_dtype_for_real(
        jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    )
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

    # Leaves live in [num_internal, total_nodes)
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return packed

    batch = max(1, int(leaf_batch_size))
    num_leaves = int(total_nodes - num_internal)
    steps = (num_leaves + batch - 1) // batch
    pad_amount = steps * batch - num_leaves
    leaf_nodes = jnp.pad(
        leaf_nodes,
        (0, pad_amount),
        mode="constant",
        constant_values=int(num_internal),
    )
    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    batch_offsets = jnp.arange(batch, dtype=INDEX_DTYPE)

    def leaf_accumulate(pos_i: Array, mass_i: Array, center_i: Array) -> Array:
        delta = pos_i - center_i
        return p2m_complex_batch(delta, mass_i, order=p)

    leaf_vm = jax.vmap(leaf_accumulate, in_axes=(0, 0, 0))

    def body(state: Array, step_idx: Array) -> tuple[Array, None]:
        start = step_idx * batch
        batch_nodes = lax.dynamic_slice_in_dim(leaf_nodes, start, batch, axis=0)
        remaining = num_leaves - start
        batch_len = jnp.minimum(batch, jnp.maximum(remaining, 0))
        valid_leaf = batch_offsets < batch_len
        safe_nodes = jnp.where(valid_leaf, batch_nodes, as_index(num_internal))

        ranges = jnp.asarray(node_ranges, dtype=INDEX_DTYPE)[safe_nodes]
        starts = ranges[:, 0]
        ends_inclusive = ranges[:, 1]
        counts = ends_inclusive - starts + 1
        particle_idx = starts[:, None] + idx[None, :]
        valid_particle = valid_leaf[:, None] & (idx[None, :] < counts[:, None])
        safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)

        pos = positions_sorted[safe_idx]
        pos = jnp.where(valid_particle[..., None], pos, 0.0)
        masses = masses_sorted[safe_idx]
        masses = jnp.where(valid_particle, masses, 0.0)

        contribs = leaf_vm(pos, masses, centers[safe_nodes])
        leaf_coeffs = jnp.sum(contribs, axis=1).astype(state.dtype)
        leaf_coeffs = enforce_conjugate_symmetry_batch(leaf_coeffs, order=p)
        current = state[safe_nodes]
        updates = jnp.where(valid_leaf[:, None], leaf_coeffs, current)
        return state.at[safe_nodes].set(updates), None

    packed, _ = lax.scan(
        body,
        packed,
        jnp.arange(steps, dtype=INDEX_DTYPE),
    )
    return packed


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "time_derivative_order",
        "max_leaf_size",
        "num_internal",
        "total_nodes",
    ),
)
def _p2m_leaves_complex_source_motion(
    node_ranges: Array,
    positions_sorted: Array,
    masses_sorted: Array,
    velocities_sorted: Array,
    centers: Array,
    *,
    order: int,
    time_derivative_order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
) -> Array:
    """Leaf source-motion P2M: d/dt[m * R(delta)] for fixed expansion centers."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    td_order = int(time_derivative_order)
    if td_order <= 0:
        raise ValueError("time_derivative_order must be positive")

    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    dtype = complex_dtype_for_real(
        jnp.result_type(
            positions_sorted.dtype,
            masses_sorted.dtype,
            velocities_sorted.dtype,
        )
    )
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return packed

    ranges = jnp.asarray(node_ranges, dtype=INDEX_DTYPE)[leaf_nodes]
    starts = ranges[:, 0]
    ends_inclusive = ranges[:, 1]
    counts = ends_inclusive - starts + 1

    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    particle_idx = starts[:, None] + idx[None, :]
    valid = idx[None, :] < counts[:, None]
    safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)

    pos = positions_sorted[safe_idx]
    pos = jnp.where(valid[..., None], pos, 0.0)
    masses = masses_sorted[safe_idx]
    masses = jnp.where(valid, masses, 0.0)
    vel = velocities_sorted[safe_idx]
    vel = jnp.where(valid[..., None], vel, 0.0)

    delta = pos - centers[leaf_nodes][:, None, :]
    part_deriv = regular_solid_harmonic_directional_derivative_order_batch(
        delta.reshape((-1, 3)),
        vel.reshape((-1, 3)),
        order=p,
        derivative_order=td_order,
    )
    part_deriv = part_deriv.reshape((delta.shape[0], delta.shape[1], coeffs))
    part_deriv = part_deriv.astype(packed.dtype)
    part_deriv = jnp.where(valid[..., None], part_deriv, 0)

    leaf_coeffs = jnp.sum(masses[..., None] * part_deriv, axis=1)
    leaf_coeffs = enforce_conjugate_symmetry_batch(leaf_coeffs, order=p)
    packed = packed.at[leaf_nodes].set(leaf_coeffs)
    return packed


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "num_internal",
        "num_levels",
        "level_batch_width",
        "rotation",
    ),
)
def _aggregate_m2m_complex_by_level(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    rotation: str,
) -> Array:
    """Upward aggregation by translating child multipoles level by level."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    if int(num_internal) <= 0:
        # Leaf-only trees have no child->parent aggregation work.
        return packed

    batch_width = int(max(level_batch_width, 1))
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    level_slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)

    def _translate_one(coeffs: Array, delta: Array) -> Array:
        return m2m_complex(coeffs, delta, order=p, rotation=rotation).astype(
            packed.dtype
        )

    translate_children = jax.vmap(
        jax.vmap(_translate_one, in_axes=(0, 0)),
        in_axes=(0, 0),
    )

    def level_body(level_rev_idx: Array, state: Array) -> Array:
        level_idx = as_index((num_levels - 2) - level_rev_idx)
        start = level_offsets[level_idx]
        end = level_offsets[level_idx + 1]
        count = end - start
        batch_nodes = lax.dynamic_slice_in_dim(
            nodes_by_level,
            start_index=start,
            slice_size=batch_width,
            axis=0,
        )
        valid = level_slot < count
        internal_valid = valid & (batch_nodes < as_index(num_internal))
        safe_nodes = jnp.where(internal_valid, batch_nodes, as_index(0))

        child_idx_pair = jnp.stack(
            [left_child[safe_nodes], right_child[safe_nodes]],
            axis=1,
        )
        child_mask = child_idx_pair >= 0
        safe_child_idx = jnp.where(child_mask, child_idx_pair, 0)
        child_coeffs = state[safe_child_idx]
        child_centers = centers[safe_child_idx]
        node_centers = centers[safe_nodes][:, None, :]
        deltas = child_centers - node_centers

        translated = translate_children(child_coeffs, deltas)
        translated = translated * child_mask[..., None]
        node_coeffs = jnp.sum(translated, axis=1, dtype=translated.dtype)
        node_coeffs = enforce_conjugate_symmetry_batch(node_coeffs, order=p)

        current = state[safe_nodes]
        updates = jnp.where(internal_valid[:, None], node_coeffs, current)
        return state.at[safe_nodes].set(updates)

    internal_level_count = max(int(num_levels) - 1, 0)
    return lax.fori_loop(0, internal_level_count, level_body, packed)


@jaxtyped(typechecker=beartype)
def prepare_solidfmm_complex_upward_sweep(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    velocities_sorted: Optional[Array] = None,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    max_leaf_size: Optional[int] = None,
    leaf_batch_size: Optional[int] = None,
    rotation: str = "cached",
    precomputed_geometry: Optional[TreeGeometry] = None,
) -> SolidFMMComplexTreeUpwardData:
    """Compute complex multipoles for every node (solidfmm basis)."""

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")

    _upward_diag(
        "geometry start "
        f"particles={int(positions_sorted.shape[0])} max_order={p} rotation={rotation}"
    )
    # Thread the known leaf cap into geometry so JIT does not pad leaf-bound
    # gathers out to ``num_particles`` for large radix trees.
    geometry = (
        precomputed_geometry
        if precomputed_geometry is not None
        else compute_tree_geometry(
            tree,
            positions_sorted,
            max_leaf_size=int(max_leaf_size) if max_leaf_size is not None else None,
        )
    )
    _upward_diag("geometry done")
    mass_moments = compute_tree_mass_moments(
        tree,
        positions_sorted,
        masses_sorted,
    )
    _upward_diag("mass moments done")

    total_nodes = int(tree.parent.shape[0])
    mode = str(center_mode).strip().lower()
    if mode == "com":
        centers = mass_moments.center_of_mass
    elif mode == "aabb":
        centers = geometry.center
    elif mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (total_nodes, 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = explicit_centers
    else:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers = jnp.asarray(centers, dtype=positions_sorted.dtype)

    if max_leaf_size is None:
        num_internal = int(tree.num_internal_nodes)
        leaf_ranges = jax.device_get(tree.node_ranges)[num_internal:]
        if leaf_ranges.shape[0] == 0:
            max_leaf_size = 0
        else:
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            max_leaf_size = int(jnp.max(counts))

    num_internal = int(tree.num_internal_nodes)
    total_nodes = int(tree.parent.shape[0])
    num_leaves = max(total_nodes - num_internal, 0)
    level_offsets = get_level_offsets(tree)
    nodes_by_level = get_nodes_by_level(tree)
    num_levels = int(level_offsets.shape[0] - 1)
    if num_levels <= 0:
        num_levels = 1
    # Keep batching shape-derived so this path remains JIT-safe under traced tree
    # builds, but use a tighter per-level bound to avoid inflating static shapes
    # with the total number of internal nodes.
    level_batch_width = max(int(num_internal), 1)
    resolved_leaf_batch_size = (
        min(num_leaves, _DEFAULT_LEAF_BATCH_SIZE)
        if leaf_batch_size is None
        else int(leaf_batch_size)
    )
    _upward_diag(
        "batch sizing "
        f"total_nodes={total_nodes} num_internal={num_internal} num_leaves={num_leaves} "
        f"resolved_leaf_batch_size={resolved_leaf_batch_size} "
        f"num_levels={num_levels} level_batch_width={level_batch_width}"
    )
    _diag_upward_stage_estimates(
        num_particles=int(positions_sorted.shape[0]),
        total_nodes=total_nodes,
        num_leaves=num_leaves,
        max_leaf_size=int(max_leaf_size),
        leaf_batch_size=resolved_leaf_batch_size,
        coeffs=sh_size(p),
        positions_dtype=positions_sorted.dtype,
        masses_dtype=masses_sorted.dtype,
    )

    _upward_diag("p2m start")
    packed = _p2m_leaves_complex(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        centers,
        order=p,
        max_leaf_size=int(max_leaf_size),
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_batch_size=resolved_leaf_batch_size,
    )
    _upward_diag(f"p2m done packed_shape={tuple(int(v) for v in packed.shape)}")

    _upward_diag("m2m start")
    packed = _aggregate_m2m_complex_by_level(
        packed,
        centers,
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE),
        jnp.asarray(level_offsets, dtype=INDEX_DTYPE),
        order=p,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        rotation=rotation,
    )
    _upward_diag("m2m done")

    source_motion_packed: Optional[Array] = None
    if velocities_sorted is not None:
        source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
            tree,
            positions_sorted,
            masses_sorted,
            velocities_sorted,
            max_order=p,
            centers=centers,
            max_leaf_size=int(max_leaf_size),
            rotation=rotation,
        )

    multipoles = SolidFMMComplexNodeMultipoleData(
        order=p,
        centers=centers,
        packed=packed,
        source_motion_packed=source_motion_packed,
    )

    return SolidFMMComplexTreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )


@jaxtyped(typechecker=beartype)
def prepare_solidfmm_complex_source_motion_multipoles(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    velocities_sorted: Array,
    *,
    max_order: int,
    centers: Array,
    time_derivative_order: int = 1,
    max_leaf_size: Optional[int] = None,
    rotation: str = "cached",
) -> Array:
    """Compute packed source-motion multipoles for fixed expansion centers."""

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")
    td_order = int(time_derivative_order)
    if td_order <= 0:
        raise ValueError("time_derivative_order must be positive")
    centers_arr = jnp.asarray(centers, dtype=positions_sorted.dtype)
    if centers_arr.shape != (int(tree.parent.shape[0]), 3):
        raise ValueError("centers must have shape (num_nodes, 3)")
    vel_sorted_arr = jnp.asarray(velocities_sorted, dtype=positions_sorted.dtype)
    if vel_sorted_arr.shape != positions_sorted.shape:
        raise ValueError(
            "velocities_sorted must have shape "
            f"{tuple(positions_sorted.shape)}, got {tuple(vel_sorted_arr.shape)}"
        )

    if max_leaf_size is None:
        num_internal = int(tree.num_internal_nodes)
        leaf_ranges = jax.device_get(tree.node_ranges)[num_internal:]
        if leaf_ranges.shape[0] == 0:
            max_leaf_size = 0
        else:
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            max_leaf_size = int(jnp.max(counts))

    num_internal = int(tree.num_internal_nodes)
    total_nodes = int(tree.parent.shape[0])
    source_motion_packed_leaf = _p2m_leaves_complex_source_motion(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        vel_sorted_arr,
        centers_arr,
        order=p,
        time_derivative_order=td_order,
        max_leaf_size=int(max_leaf_size),
        num_internal=num_internal,
        total_nodes=total_nodes,
    )
    level_offsets = get_level_offsets(tree)
    nodes_by_level = get_nodes_by_level(tree)
    num_levels = int(level_offsets.shape[0] - 1)
    if num_levels <= 0:
        num_levels = 1
    level_batch_width = max(int(num_internal), 1)

    return _aggregate_m2m_complex_by_level(
        source_motion_packed_leaf,
        centers_arr,
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE),
        jnp.asarray(level_offsets, dtype=INDEX_DTYPE),
        order=p,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        rotation=rotation,
    )
