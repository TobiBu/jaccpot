"""SolidFMM-style complex spherical-harmonics tree expansion helpers.

This module implements the *upward sweep* (P2M + M2M) for the complex
solid-harmonic basis used by the solidfmm-style backend. It is intentionally
kept separate from the Dehnen real-basis implementation.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped

from jaccpot.operators.complex_harmonics import p2m_complex_batch
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry,
    enforce_conjugate_symmetry_batch,
    m2m_complex,
)
from yggdrasil.dtypes import INDEX_DTYPE, complex_dtype_for_real
from jaccpot.operators.real_harmonics import sh_size
from yggdrasil.tree import RadixTree
from yggdrasil.geometry import TreeGeometry, compute_tree_geometry
from yggdrasil.tree_moments import TreeMassMoments, compute_tree_mass_moments


class SolidFMMComplexNodeMultipoleData(NamedTuple):
    """Packed complex multipole coefficients and their metadata."""

    order: int
    centers: Array  # (num_nodes, 3)
    packed: Array  # (num_nodes, (p+1)^2)


class SolidFMMComplexTreeUpwardData(NamedTuple):
    """Container bundling data needed for the complex upward sweep."""

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: SolidFMMComplexNodeMultipoleData


_CENTER_MODES = ("com", "aabb", "explicit")


@partial(
    jax.jit,
    static_argnames=("order", "max_leaf_size", "num_internal", "total_nodes"),
)
def _p2m_leaves_complex(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    centers: Array,
    *,
    order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
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

    ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)[leaf_nodes]
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

    def leaf_accumulate(pos_i: Array, mass_i: Array, center_i: Array) -> Array:
        delta = pos_i - center_i
        return p2m_complex_batch(delta, mass_i, order=p)

    leaf_vm = jax.vmap(leaf_accumulate, in_axes=(0, 0, 0))
    contribs = leaf_vm(pos, masses, centers[leaf_nodes])
    leaf_coeffs = jnp.sum(contribs, axis=1)
    leaf_coeffs = leaf_coeffs.astype(packed.dtype)
    leaf_coeffs = enforce_conjugate_symmetry_batch(leaf_coeffs, order=p)
    packed = packed.at[leaf_nodes].set(leaf_coeffs)

    return packed


@partial(jax.jit, static_argnames=("order", "num_internal", "rotation"))
def _aggregate_m2m_complex(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    num_internal: int,
    rotation: str,
) -> Array:
    """Upward aggregation by translating child multipoles to parent."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    if int(num_internal) <= 0:
        # Leaf-only trees have no child->parent aggregation work.
        return packed

    def translate_children(
        node_idx: Array,
        child_idx_pair: Array,
        state: Array,
    ) -> Array:
        child_mask = child_idx_pair >= 0
        safe_child_idx = jnp.where(child_mask, child_idx_pair, 0)
        child_coeffs = state[safe_child_idx]
        child_centers = centers[safe_child_idx]
        deltas = child_centers - centers[node_idx]

        def translate_one(coeffs, delta):
            return m2m_complex(coeffs, delta, order=p, rotation=rotation).astype(
                state.dtype
            )

        translated = jax.vmap(translate_one)(child_coeffs, deltas)
        translated = translated * child_mask[:, None]
        node_coeff = jnp.sum(translated, axis=0)
        return enforce_conjugate_symmetry(node_coeff, order=p)

    def body(node_idx, state):
        child_idx_pair = jnp.stack(
            [left_child[node_idx], right_child[node_idx]],
            axis=0,
        )
        node_coeff = translate_children(node_idx, child_idx_pair, state)
        return state.at[node_idx].set(node_coeff)

    import math

    max_depth = int(math.ceil(math.log2(max(num_internal + 1, 2)))) + 1

    def one_pass(state):
        return lax.fori_loop(0, num_internal, body, state)

    return lax.fori_loop(0, max_depth, lambda _, s: one_pass(s), packed)


@jaxtyped(typechecker=beartype)
def prepare_solidfmm_complex_upward_sweep(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    max_leaf_size: Optional[int] = None,
    rotation: str = "cached",
) -> SolidFMMComplexTreeUpwardData:
    """Compute complex multipoles for every node (solidfmm basis)."""

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")

    geometry = compute_tree_geometry(tree, positions_sorted)
    mass_moments = compute_tree_mass_moments(
        tree,
        positions_sorted,
        masses_sorted,
    )

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

    packed = _p2m_leaves_complex(
        tree,
        positions_sorted,
        masses_sorted,
        centers,
        order=p,
        max_leaf_size=int(max_leaf_size),
        num_internal=num_internal,
        total_nodes=total_nodes,
    )

    packed = _aggregate_m2m_complex(
        packed,
        centers,
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        order=p,
        num_internal=num_internal,
        rotation=rotation,
    )

    multipoles = SolidFMMComplexNodeMultipoleData(
        order=p,
        centers=centers,
        packed=packed,
    )

    return SolidFMMComplexTreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )
