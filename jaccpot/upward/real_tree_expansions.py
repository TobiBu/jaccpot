"""Native real (Dehnen no-sqrt2) tree upward sweep: P2M + M2M by level.

Real-basis analog of
:func:`jaccpot.upward.solidfmm_complex_tree_expansions.prepare_solidfmm_complex_upward_sweep`.
It produces per-node **real** multipole coefficients directly (no complex
intermediate), which matters for the distributed FMM: the coarse-tree seeding
``all_gather``\\s the leaf multipoles, so real coefficients halve the inter-GPU
communication versus the complex packed multipoles.

The result is mathematically identical (to machine precision) to running the
complex upward sweep and converting with
:func:`jaccpot.operators.real_harmonics.complex_to_dehnen_real_coeffs` -- that
equivalence is the correctness oracle used by the unit test. The structure
(leaf-batched P2M scan, level-by-level M2M with dead-row scatter) is a faithful
copy of the complex module; only the per-node operators (``p2m_real_direct`` /
``m2m_real``), the real dtype, and the absence of the complex-only
conjugate-symmetry fixup differ.

Center mode is COM only (what the distributed driver + fast lane use).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.tree import Tree, get_level_offsets, get_nodes_by_level
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.operators.real_harmonics import m2m_real, p2m_real_direct, sh_size

__all__ = [
    "RealNodeMultipoleData",
    "RealTreeUpwardData",
    "prepare_real_upward_sweep",
    "aggregate_m2m_real_by_level",
]

_DEFAULT_LEAF_BATCH_SIZE = 2048


class RealNodeMultipoleData(NamedTuple):
    """Packed real multipole coefficients and their metadata."""

    order: int
    centers: Array  # (num_nodes, 3)
    packed: Array  # (num_nodes, (p+1)^2), real


class RealTreeUpwardData(NamedTuple):
    """Container mirroring the complex sweep's ``.multipoles`` access path."""

    multipoles: RealNodeMultipoleData


def _p2m_leaves_real(
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
    """Leaf P2M in the Dehnen real basis (mirror of ``_p2m_leaves_complex``)."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    dtype = jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

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

    def _p2m_real_batch(deltas: Array, masses: Array) -> Array:
        # per-particle real P2M, (K,3),(K,) -> (K, C)
        return jax.vmap(lambda d, m: p2m_real_direct(d, m, order=p))(deltas, masses)

    def leaf_accumulate(pos_i: Array, mass_i: Array, center_i: Array) -> Array:
        delta = pos_i - center_i
        return _p2m_real_batch(delta, mass_i)

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
        current = state[safe_nodes]
        updates = jnp.where(valid_leaf[:, None], leaf_coeffs, current)
        return state.at[safe_nodes].set(updates), None

    packed, _ = lax.scan(body, packed, jnp.arange(steps, dtype=INDEX_DTYPE))
    return packed


def aggregate_m2m_real_by_level(
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
) -> Array:
    """Real-basis upward M2M aggregation by level (mirror of the complex one).

    Reused for both the local tree upward sweep and the distributed coarse tree.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    if int(num_internal) <= 0:
        return packed

    batch_width = int(max(level_batch_width, 1))
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    level_slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)

    def _translate_one(coeffs: Array, delta: Array) -> Array:
        return m2m_real(coeffs, delta, order=p).astype(packed.dtype)

    translate_children = jax.vmap(
        jax.vmap(_translate_one, in_axes=(0, 0)),
        in_axes=(0, 0),
    )

    # Dead row for padding/invalid scatters (see the complex version's note: a
    # duplicate scatter onto a real node id would clobber a genuine M2M result).
    dead_row = as_index(packed.shape[0])
    packed_ext = jnp.concatenate(
        [packed, jnp.zeros((1,) + tuple(packed.shape[1:]), dtype=packed.dtype)],
        axis=0,
    )

    def level_body(level_rev_idx: Array, state: Array) -> Array:
        level_idx = as_index((num_levels - 2) - level_rev_idx)
        start = level_offsets[level_idx]
        end = level_offsets[level_idx + 1]
        count = end - start
        batch_nodes = lax.dynamic_slice_in_dim(
            nodes_by_level, start_index=start, slice_size=batch_width, axis=0
        )
        valid = level_slot < count
        internal_valid = valid & (batch_nodes < as_index(num_internal))
        gather_nodes = jnp.where(internal_valid, batch_nodes, as_index(0))
        scatter_nodes = jnp.where(internal_valid, batch_nodes, dead_row)

        child_idx_pair = jnp.stack(
            [left_child[gather_nodes], right_child[gather_nodes]], axis=1
        )
        child_mask = child_idx_pair >= 0
        safe_child_idx = jnp.where(child_mask, child_idx_pair, 0)
        child_coeffs = state[safe_child_idx]
        child_centers = centers[safe_child_idx]
        node_centers = centers[gather_nodes][:, None, :]
        deltas = child_centers - node_centers

        translated = translate_children(child_coeffs, deltas)
        translated = translated * child_mask[..., None]
        node_coeffs = jnp.sum(translated, axis=1, dtype=translated.dtype)

        return state.at[scatter_nodes].set(node_coeffs)

    internal_level_count = max(int(num_levels) - 1, 0)
    result = lax.fori_loop(0, internal_level_count, level_body, packed_ext)
    return result[: packed.shape[0]]


def prepare_real_upward_sweep(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    max_leaf_size: int,
    leaf_batch_size: Optional[int] = None,
    static_num_levels: Optional[int] = None,
) -> RealTreeUpwardData:
    """Native real-basis upward sweep (COM centers): P2M leaves + M2M by level.

    ``max_leaf_size`` is required (the distributed/fast-lane callers know it): it
    keeps the leaf-bound gathers from padding out to ``num_particles`` and avoids
    a concrete device_get under trace.
    """

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")

    mass_moments = compute_tree_mass_moments(tree, positions_sorted, masses_sorted)
    centers = jnp.asarray(mass_moments.center_of_mass, dtype=positions_sorted.dtype)

    total_nodes = int(jnp.asarray(tree.parent).shape[0])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    num_leaves = max(total_nodes - num_internal, 0)

    level_offsets = get_level_offsets(tree)
    nodes_by_level = get_nodes_by_level(tree)
    if static_num_levels is not None:
        num_levels = max(int(static_num_levels), 1)
    else:
        num_levels = int(level_offsets.shape[0] - 1)
        if num_levels <= 0:
            num_levels = 1
    level_batch_width = max(int(num_internal), 1)
    resolved_leaf_batch_size = (
        min(num_leaves, _DEFAULT_LEAF_BATCH_SIZE)
        if leaf_batch_size is None
        else int(leaf_batch_size)
    )

    packed = _p2m_leaves_real(
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
    packed = aggregate_m2m_real_by_level(
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
    )
    return RealTreeUpwardData(
        multipoles=RealNodeMultipoleData(order=p, centers=centers, packed=packed)
    )
