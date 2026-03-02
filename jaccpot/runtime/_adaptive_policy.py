"""Solver-owned adaptive traversal policy helpers."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from yggdrax.tree import Tree

from jaccpot.upward.tree_expansions import TreeUpwardData

_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2

_ERROR_MODEL_TAIL_PROXY = 0
_ERROR_MODEL_DEHNEN_DEGREE = 1
_ERROR_MODEL_DEHNEN_PAPER = 2


class AdaptivePolicyState(NamedTuple):
    """Solver-owned per-node summaries used by adaptive traversal policies."""

    source_error_proxy_by_order: Array
    source_degree_power: Array
    source_dehnen_power: Array
    source_mass: Array
    source_mac_center: Array
    target_mac_center: Array
    source_radius_bound: Array
    target_radius_bound: Array
    target_accept_threshold: Array
    order_tags: Array
    order_values: Array
    dehnen_binomial_by_order: Array
    relaxed_theta_sq: Array
    error_model_code: Array


def adaptive_policy_tolerance(
    *, theta: float, p_gears: tuple[int, ...], dtype: object
) -> Array:
    """Return a conservative solver-side adaptive tolerance derived from ``theta``."""

    if len(p_gears) == 0:
        raise ValueError("adaptive policy tolerance requires non-empty p_gears")
    return jnp.asarray(float(theta) ** (max(int(v) for v in p_gears) + 2), dtype=dtype)


def _packed_total_order(multipole_packed: Array) -> int:
    packed = jnp.asarray(multipole_packed)
    return int(round(np.sqrt(int(packed.shape[1])) - 1))


def source_power_by_degree_from_multipoles(*, multipole_packed: Array) -> Array:
    """Return per-node multipole power grouped by spherical-harmonic degree."""

    packed = jnp.asarray(multipole_packed)
    total_p = _packed_total_order(packed)
    magnitudes_sq = jnp.square(jnp.abs(packed))
    powers: list[Array] = []
    for ell in range(total_p + 1):
        start = ell * ell
        stop = (ell + 1) * (ell + 1)
        powers.append(jnp.sum(magnitudes_sq[:, start:stop], axis=1))
    return jnp.stack(powers, axis=1)


def dehnen_multipole_power_by_degree(*, multipole_packed: Array) -> Array:
    """Return Dehnen's exact per-degree source power ``P_n`` from packed moments.

    For the real-Dehnen basis used by the real SH runtime, equation (12) maps
    directly onto the packed coefficients ``M_n^m``:

    ``P_n^2 = sum_m (n-m)! (n+m)! |M_n^m|^2``.
    """

    packed = jnp.asarray(multipole_packed)
    total_p = _packed_total_order(packed)
    dtype = packed.real.dtype if jnp.iscomplexobj(packed) else packed.dtype
    factorial = jnp.exp(jax.lax.lgamma(jnp.arange(2 * total_p + 1, dtype=dtype) + 1.0))
    powers: list[Array] = []
    for ell in range(total_p + 1):
        start = ell * ell
        stop = (ell + 1) * (ell + 1)
        degree_slice = packed[:, start:stop]
        m_vals = jnp.arange(-ell, ell + 1, dtype=jnp.int32)
        weights = factorial[ell - m_vals] * factorial[ell + m_vals]
        weighted_sq = jnp.square(jnp.abs(degree_slice)) * weights[None, :]
        powers.append(jnp.sqrt(jnp.sum(weighted_sq, axis=1)))
    return jnp.stack(powers, axis=1)


def source_error_proxy_by_order_from_degree_power(
    *,
    degree_power: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Return the residual tail proxy for each candidate order from degree power."""

    power = jnp.asarray(degree_power)
    if len(p_gears) == 0:
        return jnp.zeros((power.shape[0], 0), dtype=power.dtype)
    total_p = int(power.shape[1] - 1)
    tails: list[Array] = []
    for p_gear in p_gears:
        p_clip = int(max(0, min(int(p_gear), total_p)))
        tail_power = jnp.sum(power[:, p_clip + 1 :], axis=1)
        tails.append(jnp.sqrt(tail_power))
    return jnp.stack(tails, axis=1)


def dehnen_like_pair_error_by_order_from_degree_power(
    *,
    degree_power: Array,
    opening: Array,
    order_values: Array,
) -> Array:
    """Return a Dehnen-style degree-weighted pair error estimate by order."""

    power = jnp.asarray(degree_power)
    opening_arr = jnp.asarray(opening, dtype=power.dtype)
    if opening_arr.ndim == 0:
        opening_arr = opening_arr[None]
    opening_arr = jnp.clip(opening_arr, 0.0, 1.0)
    order_arr = jnp.asarray(order_values, dtype=jnp.int32)
    if order_arr.ndim == 0:
        order_arr = order_arr[None]
    if int(power.shape[0]) != int(opening_arr.shape[0]):
        raise ValueError(
            "degree_power and opening must have matching leading dimensions"
        )
    if int(order_arr.shape[0]) == 0:
        return jnp.zeros((opening_arr.shape[0], 0), dtype=power.dtype)
    total_p = int(power.shape[1] - 1)
    degree_idx = jnp.arange(total_p + 1, dtype=jnp.int32)
    opening_weights = jnp.power(
        opening_arr[:, None],
        degree_idx[None, :].astype(power.dtype) + 2.0,
    )
    weighted_power = power * opening_weights
    include_mask = degree_idx[None, None, :] > order_arr[None, :, None]
    tail_power = jnp.sum(
        weighted_power[:, None, :] * include_mask.astype(power.dtype), axis=2
    )
    return jnp.sqrt(jnp.maximum(tail_power, jnp.asarray(0.0, dtype=power.dtype)))


def dehnen_paper_pair_error_by_order(
    *,
    source_power: Array,
    source_mass: Array,
    source_radius: Array,
    target_radius: Array,
    distance: Array,
    order_values: Array,
    binomial_by_order: Array,
) -> Array:
    """Return Dehnen's equation (15) error estimate by candidate order."""

    power = jnp.asarray(source_power)
    mass = jnp.asarray(source_mass, dtype=power.dtype)
    rho_z = jnp.asarray(source_radius, dtype=power.dtype)
    rho_s = jnp.asarray(target_radius, dtype=power.dtype)
    r = jnp.asarray(distance, dtype=power.dtype)
    order_arr = jnp.asarray(order_values, dtype=jnp.int32)
    if order_arr.ndim == 0:
        order_arr = order_arr[None]
    if int(order_arr.shape[0]) == 0:
        return jnp.zeros((power.shape[0], 0), dtype=power.dtype)
    tiny = jnp.asarray(1e-24, dtype=power.dtype)
    safe_mass = jnp.maximum(jnp.abs(mass), tiny)
    safe_r = jnp.maximum(r, tiny)
    degree_idx = jnp.arange(power.shape[1], dtype=jnp.int32)
    exponent = jnp.maximum(order_arr[:, None] - degree_idx[None, :], 0)
    rho_factor = jnp.power(
        rho_s[:, None, None], exponent[None, :, :].astype(power.dtype)
    )
    include = (degree_idx[None, :] <= order_arr[:, None]).astype(power.dtype)
    e_terms = (
        jnp.asarray(binomial_by_order, dtype=power.dtype)[None, :, :]
        * include[None, :, :]
        * power[:, None, :]
        * rho_factor
    )
    r_pow = jnp.power(safe_r[:, None], order_arr[None, :].astype(power.dtype))
    e_basic = jnp.sum(e_terms, axis=2) / (safe_mass[:, None] * r_pow)
    improvement = (
        jnp.asarray(8.0, dtype=power.dtype)
        * jnp.maximum(rho_z, rho_s)
        / jnp.maximum(rho_z + rho_s, tiny)
    )
    return improvement[:, None] * e_basic


def source_error_proxy_by_order_from_multipoles(
    *,
    multipole_packed: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Compute a conservative per-node residual proxy for each candidate order."""

    degree_power = source_power_by_degree_from_multipoles(
        multipole_packed=multipole_packed,
    )
    return source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=p_gears,
    )


def compute_node_force_scale_from_sorted_acc(
    *,
    node_ranges: Array,
    accelerations_sorted: Array,
    reduction: str = "max",
) -> Array:
    """Estimate per-node force scales from sorted per-particle accelerations."""

    node_ranges_np = np.asarray(node_ranges, dtype=np.int64)
    acc_sorted_np = np.asarray(accelerations_sorted)
    mag = np.linalg.norm(acc_sorted_np, axis=1)
    reduction_norm = str(reduction).strip().lower()
    if reduction_norm not in ("max", "min"):
        raise ValueError("reduction must be 'max' or 'min'")
    force_scale = np.zeros((node_ranges_np.shape[0],), dtype=mag.dtype)
    for idx, (start, end) in enumerate(node_ranges_np):
        s = int(start)
        e = int(end)
        if e < s:
            continue
        values = mag[s : e + 1]
        force_scale[idx] = (
            float(np.min(values)) if reduction_norm == "min" else float(np.max(values))
        )
    return jnp.asarray(force_scale, dtype=accelerations_sorted.dtype)


def _sphere_from_support(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the exact sphere defined by up to four support points."""

    pts = np.asarray(points, dtype=np.float64)
    count = int(pts.shape[0])
    if count == 0:
        return np.zeros((3,), dtype=np.float64), -1.0
    if count == 1:
        return pts[0], 0.0
    if count == 2:
        center = 0.5 * (pts[0] + pts[1])
        return center, float(np.linalg.norm(pts[0] - center))
    if count == 3:
        a, b, c = pts
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        denom = 2.0 * float(np.dot(cross, cross))
        if denom <= 1e-24:
            best_center = pts[0]
            best_radius = float("inf")
            for i in range(3):
                for j in range(i + 1, 3):
                    center, radius = _sphere_from_support(pts[[i, j]])
                    if (
                        np.all(
                            np.linalg.norm(pts - center[None, :], axis=1)
                            <= radius + 1e-10
                        )
                        and radius < best_radius
                    ):
                        best_center = center
                        best_radius = radius
            return best_center, best_radius
        center = (
            a
            + (
                np.cross(cross, ab) * float(np.dot(ac, ac))
                + np.cross(ac, cross) * float(np.dot(ab, ab))
            )
            / denom
        )
        return center, float(np.linalg.norm(a - center))
    if count == 4:
        p0 = pts[0]
        A = 2.0 * (pts[1:] - p0)
        b = np.sum(pts[1:] * pts[1:], axis=1) - float(np.dot(p0, p0))
        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            best_center = pts[0]
            best_radius = float("inf")
            from itertools import combinations

            for size in (2, 3):
                for combo in combinations(range(4), size):
                    center, radius = _sphere_from_support(pts[list(combo)])
                    if (
                        np.all(
                            np.linalg.norm(pts - center[None, :], axis=1)
                            <= radius + 1e-10
                        )
                        and radius < best_radius
                    ):
                        best_center = center
                        best_radius = radius
            return best_center, best_radius
        return center, float(np.linalg.norm(p0 - center))
    raise ValueError("support must contain at most four points")


def _point_in_sphere(
    point: np.ndarray, center: np.ndarray, radius: float, tol: float = 1e-10
) -> bool:
    if radius < 0.0:
        return False
    return float(np.linalg.norm(point - center)) <= radius + tol


def _smallest_enclosing_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the exact smallest enclosing sphere for a 3D point set."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64), 0.0
    center = pts[0]
    radius = 0.0
    for i in range(pts.shape[0]):
        p = pts[i]
        if _point_in_sphere(p, center, radius):
            continue
        center = p
        radius = 0.0
        for j in range(i):
            q = pts[j]
            if _point_in_sphere(q, center, radius):
                continue
            center, radius = _sphere_from_support(np.stack([p, q], axis=0))
            for k in range(j):
                r = pts[k]
                if _point_in_sphere(r, center, radius):
                    continue
                center, radius = _sphere_from_support(np.stack([p, q, r], axis=0))
                for l in range(k):
                    s = pts[l]
                    if _point_in_sphere(s, center, radius):
                        continue
                    center, radius = _sphere_from_support(
                        np.stack([p, q, r, s], axis=0)
                    )
    return center, radius


def compute_smallest_enclosing_sphere_geometry(
    *, node_ranges: Array, positions_sorted: Array
) -> tuple[Array, Array]:
    """Return exact SES centres and radii for each node range."""

    ranges = np.asarray(node_ranges, dtype=np.int64)
    pos = np.asarray(positions_sorted, dtype=np.float64)
    centers = np.zeros((ranges.shape[0], pos.shape[1]), dtype=np.float64)
    radii = np.zeros((ranges.shape[0],), dtype=np.float64)
    for idx, (start, end) in enumerate(ranges):
        s = int(start)
        e = int(end)
        if e < s:
            continue
        center, radius = _smallest_enclosing_sphere(pos[s : e + 1])
        centers[idx] = center
        radii[idx] = radius
    return (
        jnp.asarray(centers, dtype=positions_sorted.dtype),
        jnp.asarray(radii, dtype=positions_sorted.dtype),
    )


def compute_leaf_enclosing_sphere_geometry(
    *, tree: Tree, positions_sorted: Array
) -> tuple[Array, Array]:
    """Return exact SES centres and radii for leaf nodes only."""

    node_ranges = np.asarray(tree.node_ranges, dtype=np.int64)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(tree.num_internal_nodes)
    centers = np.zeros((num_nodes, positions_sorted.shape[1]), dtype=np.float64)
    radii = np.zeros((num_nodes,), dtype=np.float64)
    if num_internal > 0:
        leaf_ranges = node_ranges[num_internal:]
    else:
        leaf_ranges = node_ranges
    pos = np.asarray(positions_sorted, dtype=np.float64)
    for leaf_offset, (start, end) in enumerate(leaf_ranges):
        s = int(start)
        e = int(end)
        if e < s:
            continue
        center, radius = _smallest_enclosing_sphere(pos[s : e + 1])
        node_idx = num_internal + leaf_offset
        centers[node_idx] = center
        radii[node_idx] = radius
    return (
        jnp.asarray(centers, dtype=positions_sorted.dtype),
        jnp.asarray(radii, dtype=positions_sorted.dtype),
    )


@jax.jit
def _batched_ritter_leaf_spheres(
    leaf_points: Array, leaf_valid: Array
) -> tuple[Array, Array]:
    """Return approximate bounding spheres for padded leaf particle blocks."""

    points = jnp.asarray(leaf_points)
    valid = jnp.asarray(leaf_valid, dtype=jnp.bool_)
    dtype = points.dtype
    valid_f = valid.astype(dtype)
    counts = jnp.sum(valid_f, axis=1)
    default_center = jnp.sum(points * valid_f[..., None], axis=1) / jnp.maximum(
        counts[:, None], jnp.asarray(1.0, dtype=dtype)
    )

    def leaf_fn(
        pts: Array, mask: Array, count: Array, fallback_center: Array
    ) -> tuple[Array, Array]:
        count_i = count.astype(jnp.int32)

        def no_points(_: None) -> tuple[Array, Array]:
            return fallback_center, jnp.asarray(0.0, dtype=dtype)

        def with_points(_: None) -> tuple[Array, Array]:
            first_idx = jnp.argmax(mask.astype(jnp.int32))
            p0 = pts[first_idx]
            d0 = jnp.where(
                mask,
                jnp.sum(jnp.square(pts - p0[None, :]), axis=1),
                -jnp.ones(mask.shape, dtype=dtype),
            )
            p1_idx = jnp.argmax(d0)
            p1 = pts[p1_idx]
            d1 = jnp.where(
                mask,
                jnp.sum(jnp.square(pts - p1[None, :]), axis=1),
                -jnp.ones(mask.shape, dtype=dtype),
            )
            p2_idx = jnp.argmax(d1)
            p2 = pts[p2_idx]
            center0 = 0.5 * (p1 + p2)
            radius0 = jnp.linalg.norm(p2 - center0)

            def body(i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
                center, radius = state
                point = pts[i]
                is_valid = mask[i]
                delta = point - center
                dist = jnp.linalg.norm(delta)
                expand = is_valid & (dist > radius)
                new_radius = 0.5 * (radius + dist)
                shift = jnp.where(
                    dist > jnp.asarray(1e-24, dtype=dtype),
                    ((new_radius - radius) / dist) * delta,
                    jnp.zeros_like(delta),
                )
                center = jnp.where(expand, center + shift, center)
                radius = jnp.where(expand, new_radius, radius)
                return center, radius

            return jax.lax.fori_loop(0, pts.shape[0], body, (center0, radius0))

        return jax.lax.cond(count_i <= 0, no_points, with_points, operand=None)

    centers, radii = jax.vmap(leaf_fn)(points, valid, counts, default_center)
    return centers, radii


def compute_leaf_ritter_sphere_geometry(
    *, tree: Tree, positions_sorted: Array
) -> tuple[Array, Array]:
    """Return fast approximate leaf spheres using a batched JAX Ritter pass."""

    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(tree.num_internal_nodes)
    centers = jnp.zeros(
        (num_nodes, positions_sorted.shape[1]), dtype=positions_sorted.dtype
    )
    radii = jnp.zeros((num_nodes,), dtype=positions_sorted.dtype)
    leaf_ranges = node_ranges[num_internal:] if num_internal > 0 else node_ranges
    if int(leaf_ranges.shape[0]) == 0:
        return centers, radii
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf = int(jnp.max(counts)) if int(counts.shape[0]) > 0 else 0
    idx = jnp.arange(max_leaf, dtype=jnp.int32)
    particle_idx = leaf_ranges[:, 0:1] + idx[None, :]
    valid = idx[None, :] < counts[:, None]
    safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)
    leaf_points = positions_sorted[safe_idx]
    leaf_points = jnp.where(valid[..., None], leaf_points, 0.0)
    leaf_centers, leaf_radii = _batched_ritter_leaf_spheres(leaf_points, valid)
    leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=jnp.int32)
    centers = centers.at[leaf_nodes].set(leaf_centers.astype(positions_sorted.dtype))
    radii = radii.at[leaf_nodes].set(leaf_radii.astype(positions_sorted.dtype))
    return centers, radii


def merge_bounding_spheres(
    center_a: Array, radius_a: Array, center_b: Array, radius_b: Array
) -> tuple[Array, Array]:
    """Return the minimal sphere containing two spheres."""

    center_a = jnp.asarray(center_a)
    center_b = jnp.asarray(center_b, dtype=center_a.dtype)
    radius_a = jnp.asarray(radius_a, dtype=center_a.dtype)
    radius_b = jnp.asarray(radius_b, dtype=center_a.dtype)
    delta = center_b - center_a
    tiny = jnp.asarray(1e-24, dtype=center_a.dtype)
    dist = jnp.maximum(jnp.linalg.norm(delta), tiny)
    a_contains_b = radius_a >= dist + radius_b
    b_contains_a = radius_b >= dist + radius_a
    merged_radius = 0.5 * (dist + radius_a + radius_b)
    shift = (
        ((merged_radius - radius_a) / dist)[:, None]
        if delta.ndim == 2
        else ((merged_radius - radius_a) / dist) * delta
    )
    if delta.ndim == 1:
        merged_center = center_a + ((merged_radius - radius_a) / dist) * delta
        center = jnp.where(a_contains_b[..., None], center_a, merged_center)
        center = jnp.where(b_contains_a[..., None], center_b, center)
    else:
        merged_center = center_a + ((merged_radius - radius_a) / dist)[:, None] * delta
        center = jnp.where(a_contains_b[:, None], center_a, merged_center)
        center = jnp.where(b_contains_a[:, None], center_b, center)
    radius = jnp.where(a_contains_b, radius_a, merged_radius)
    radius = jnp.where(b_contains_a, radius_b, radius)
    return center, radius


def compute_tree_merged_sphere_geometry(
    *, tree: Tree, positions_sorted: Array, leaf_mode: str = "exact"
) -> tuple[Array, Array]:
    """Return node spheres from leaf spheres and JAX upward merges."""

    leaf_mode_norm = str(leaf_mode).strip().lower()
    if leaf_mode_norm == "exact":
        centers, radii = compute_leaf_enclosing_sphere_geometry(
            tree=tree, positions_sorted=positions_sorted
        )
    elif leaf_mode_norm == "approx":
        centers, radii = compute_leaf_ritter_sphere_geometry(
            tree=tree, positions_sorted=positions_sorted
        )
    else:
        raise ValueError("leaf_mode must be 'exact' or 'approx'")
    num_internal = int(tree.num_internal_nodes)
    if num_internal == 0:
        return centers, radii
    left_child = jnp.asarray(tree.left_child, dtype=jnp.int32)
    right_child = jnp.asarray(tree.right_child, dtype=jnp.int32)

    def body(iter_idx: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        center_state, radius_state = state
        node_idx = num_internal - 1 - iter_idx
        left_idx = left_child[node_idx]
        right_idx = right_child[node_idx]

        left_center = center_state[left_idx]
        left_radius = radius_state[left_idx]

        def merge_right(_: None) -> tuple[Array, Array]:
            right_center = center_state[right_idx]
            right_radius = radius_state[right_idx]
            return merge_bounding_spheres(
                left_center, left_radius, right_center, right_radius
            )

        merged_center, merged_radius = jax.lax.cond(
            right_idx >= 0,
            merge_right,
            lambda _: (left_center, left_radius),
            operand=None,
        )
        center_state = center_state.at[node_idx].set(merged_center)
        radius_state = radius_state.at[node_idx].set(merged_radius)
        return center_state, radius_state

    return jax.lax.fori_loop(
        0,
        num_internal,
        body,
        (centers, radii),
    )


def resolve_dehnen_geometry(
    *,
    geometry_mode: Literal["exact", "tree", "tree_approx", "runtime"],
    tree: Tree,
    positions_sorted: Array,
    upward: TreeUpwardData,
    dtype: Array,
) -> tuple[Array, Array]:
    """Return MAC centres and radii for the requested Dehnen geometry mode."""

    mode = str(geometry_mode).strip().lower()
    if mode == "exact":
        mac_centers, radius_bound = compute_smallest_enclosing_sphere_geometry(
            node_ranges=tree.node_ranges,
            positions_sorted=positions_sorted,
        )
    elif mode == "tree":
        mac_centers, radius_bound = compute_tree_merged_sphere_geometry(
            tree=tree,
            positions_sorted=positions_sorted,
            leaf_mode="exact",
        )
    elif mode == "tree_approx":
        mac_centers, radius_bound = compute_tree_merged_sphere_geometry(
            tree=tree,
            positions_sorted=positions_sorted,
            leaf_mode="approx",
        )
    elif mode == "runtime":
        expansion_centers = jnp.asarray(upward.multipoles.centers, dtype=dtype)
        geometry_centers = jnp.asarray(upward.geometry.center, dtype=dtype)
        geometry_radius = jnp.asarray(upward.geometry.radius, dtype=dtype)
        center_offset = jnp.linalg.norm(expansion_centers - geometry_centers, axis=1)
        radius_bound = geometry_radius + center_offset
        mac_centers = geometry_centers
    else:
        raise ValueError(
            "dehnen_geometry_mode must be 'exact', 'tree', 'tree_approx', or 'runtime'"
        )
    return jnp.asarray(mac_centers, dtype=dtype), jnp.asarray(radius_bound, dtype=dtype)


def _dehnen_binomial_matrix(
    *, p_gears: tuple[int, ...], total_p: int, dtype: Array
) -> Array:
    rows = np.zeros((len(p_gears), total_p + 1), dtype=np.asarray(dtype).dtype)
    for idx, p_val in enumerate(tuple(int(v) for v in p_gears)):
        for ell in range(min(int(p_val), total_p) + 1):
            rows[idx, ell] = float(math.comb(int(p_val), ell))
    return jnp.asarray(rows, dtype=dtype)


def build_adaptive_policy_state(
    *,
    upward: TreeUpwardData,
    tree: Tree,
    positions_sorted: Array,
    p_gears: tuple[int, ...],
    force_scale_nodes: Optional[Array],
    eps: Array,
    theta: Array,
    error_model_code: Array,
    dehnen_geometry_mode: str = "exact",
) -> AdaptivePolicyState:
    """Build the solver-owned adaptive traversal state from upward data."""

    if len(p_gears) == 0:
        raise ValueError("adaptive policy state requires non-empty p_gears")
    packed = upward.multipoles.packed
    degree_power = source_power_by_degree_from_multipoles(multipole_packed=packed)
    dehnen_power = dehnen_multipole_power_by_degree(multipole_packed=packed)
    error_proxy = source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=p_gears,
    )
    dtype = error_proxy.dtype
    error_model_code_arr = jnp.asarray(error_model_code, dtype=jnp.int32)
    exact_dehnen = bool(int(error_model_code_arr) == _ERROR_MODEL_DEHNEN_PAPER)
    if force_scale_nodes is None:
        target_force_scale = jnp.ones((error_proxy.shape[0],), dtype=dtype)
    else:
        target_force_scale = jnp.asarray(force_scale_nodes, dtype=dtype)
        if int(target_force_scale.shape[0]) != int(error_proxy.shape[0]):
            raise ValueError("force_scale_nodes length must match number of nodes")
        if not exact_dehnen:
            scale_norm = jnp.maximum(
                jnp.max(target_force_scale),
                jnp.asarray(1.0, dtype=dtype),
            )
            target_force_scale = target_force_scale / scale_norm
    target_accept_threshold = jnp.maximum(
        jnp.asarray(eps, dtype=dtype) * target_force_scale,
        jnp.asarray(1e-24, dtype=dtype),
    )
    order_tags = jnp.arange(len(p_gears), dtype=jnp.int32)
    order_values = jnp.asarray(tuple(int(v) for v in p_gears), dtype=jnp.int32)
    theta_arr = jnp.asarray(theta, dtype=dtype)
    relaxed_theta = jnp.minimum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(1.5, dtype=dtype) * theta_arr,
    )
    total_p = int(dehnen_power.shape[1] - 1)
    exact_dehnen_geometry = bool(int(error_model_code_arr) == _ERROR_MODEL_DEHNEN_PAPER)
    if exact_dehnen_geometry:
        mac_centers, radius_bound = resolve_dehnen_geometry(
            geometry_mode=dehnen_geometry_mode,
            tree=tree,
            positions_sorted=positions_sorted,
            upward=upward,
            dtype=dtype,
        )
    else:
        mac_centers, radius_bound = resolve_dehnen_geometry(
            geometry_mode="runtime",
            tree=tree,
            positions_sorted=positions_sorted,
            upward=upward,
            dtype=dtype,
        )
    source_mass = jnp.maximum(
        jnp.abs(jnp.asarray(upward.mass_moments.mass, dtype=dtype)),
        jnp.asarray(1e-24, dtype=dtype),
    )
    return AdaptivePolicyState(
        source_error_proxy_by_order=error_proxy,
        source_degree_power=degree_power,
        source_dehnen_power=dehnen_power,
        source_mass=source_mass,
        source_mac_center=mac_centers,
        target_mac_center=mac_centers,
        source_radius_bound=radius_bound,
        target_radius_bound=radius_bound,
        target_accept_threshold=target_accept_threshold,
        order_tags=order_tags,
        order_values=order_values,
        dehnen_binomial_by_order=_dehnen_binomial_matrix(
            p_gears=p_gears,
            total_p=total_p,
            dtype=dtype,
        ),
        relaxed_theta_sq=jnp.square(relaxed_theta),
        error_model_code=error_model_code_arr,
    )


def _compute_passes_for_error_model(
    *,
    policy_state: AdaptivePolicyState,
    source_proxy: Array,
    source_degree_power: Array,
    source_dehnen_power: Array,
    source_mass: Array,
    source_radius: Array,
    target_radius: Array,
    target_threshold: Array,
    opening: Array,
    extent_sum_sq: Array,
    safe_dist_sq: Array,
    paper_distance: Array,
) -> Array:
    """Return per-order pass decisions for the configured adaptive error model."""

    def _tail_proxy(_: None) -> Array:
        return (
            jnp.square(source_proxy) * extent_sum_sq[:, None]
            < jnp.square(target_threshold)[:, None] * safe_dist_sq[:, None]
        )

    def _dehnen_degree(_: None) -> Array:
        pair_error = dehnen_like_pair_error_by_order_from_degree_power(
            degree_power=source_degree_power,
            opening=opening,
            order_values=policy_state.order_values,
        )
        return pair_error < target_threshold[:, None]

    def _dehnen_paper(_: None) -> Array:
        pair_error = dehnen_paper_pair_error_by_order(
            source_power=source_dehnen_power,
            source_mass=source_mass,
            source_radius=source_radius,
            target_radius=target_radius,
            distance=paper_distance,
            order_values=policy_state.order_values,
            binomial_by_order=policy_state.dehnen_binomial_by_order,
        )
        est_force_error = (
            pair_error
            * source_mass[:, None]
            / jnp.maximum(
                jnp.square(paper_distance[:, None]),
                jnp.asarray(1e-24, dtype=pair_error.dtype),
            )
        )
        convergent = (source_radius + target_radius) < paper_distance
        return convergent[:, None] & (est_force_error < target_threshold[:, None])

    return jax.lax.switch(
        jnp.asarray(policy_state.error_model_code, dtype=jnp.int32),
        (_tail_proxy, _dehnen_degree, _dehnen_paper),
        operand=None,
    )


def adaptive_pair_policy(
    policy_state: AdaptivePolicyState, **pair_data: Array
) -> tuple[Array, Array]:
    """Return traversal actions and order tags from solver-owned adaptive state."""

    valid_pairs = jnp.asarray(pair_data["valid_pairs"], dtype=jnp.bool_)
    mac_ok = jnp.asarray(pair_data["mac_ok"], dtype=jnp.bool_)
    different_nodes = jnp.asarray(pair_data["different_nodes"], dtype=jnp.bool_)
    target_leaf = jnp.asarray(pair_data["target_leaf"], dtype=jnp.bool_)
    source_leaf = jnp.asarray(pair_data["source_leaf"], dtype=jnp.bool_)
    target_nodes = jnp.asarray(pair_data["target_nodes"], dtype=jnp.int32)
    source_nodes = jnp.asarray(pair_data["source_nodes"], dtype=jnp.int32)
    dist_sq = jnp.asarray(pair_data["dist_sq"])
    extent_target = jnp.asarray(pair_data["extent_target"], dtype=dist_sq.dtype)
    extent_source = jnp.asarray(pair_data["extent_source"], dtype=dist_sq.dtype)

    safe_targets = jnp.where(valid_pairs, target_nodes, 0)
    safe_sources = jnp.where(valid_pairs, source_nodes, 0)
    safe_dist_sq = jnp.maximum(dist_sq, jnp.asarray(1e-24, dtype=dist_sq.dtype))
    extent_sum_sq = jnp.square(extent_target + extent_source)
    distance = jnp.sqrt(safe_dist_sq)

    source_proxy = jnp.asarray(policy_state.source_error_proxy_by_order)[
        safe_sources, :
    ]
    source_degree_power = jnp.asarray(policy_state.source_degree_power)[safe_sources, :]
    source_dehnen_power = jnp.asarray(policy_state.source_dehnen_power)[safe_sources, :]
    source_mass = jnp.asarray(policy_state.source_mass)[safe_sources]
    source_radius = jnp.asarray(policy_state.source_radius_bound)[safe_sources]
    target_radius = jnp.asarray(policy_state.target_radius_bound)[safe_targets]
    source_mac_center = jnp.asarray(policy_state.source_mac_center)[safe_sources]
    target_mac_center = jnp.asarray(policy_state.target_mac_center)[safe_targets]
    paper_distance = jnp.maximum(
        jnp.linalg.norm(source_mac_center - target_mac_center, axis=1),
        jnp.asarray(1e-24, dtype=dist_sq.dtype),
    )
    target_threshold = jnp.asarray(policy_state.target_accept_threshold)[safe_targets]
    opening = jnp.sqrt(extent_sum_sq / safe_dist_sq)
    passes = _compute_passes_for_error_model(
        policy_state=policy_state,
        source_proxy=source_proxy,
        source_degree_power=source_degree_power,
        source_dehnen_power=source_dehnen_power,
        source_mass=source_mass,
        source_radius=source_radius,
        target_radius=target_radius,
        target_threshold=target_threshold,
        opening=opening,
        extent_sum_sq=extent_sum_sq,
        safe_dist_sq=safe_dist_sq,
        paper_distance=paper_distance,
    )
    pass_any = jnp.any(passes, axis=1)
    highest_order_pass = passes[:, -1]
    allow_solver_override = (~target_leaf) | (~source_leaf)
    relaxed_mac_ok = extent_sum_sq <= policy_state.relaxed_theta_sq * safe_dist_sq

    order_tags = jnp.asarray(policy_state.order_tags, dtype=jnp.int32)
    required_idx = jnp.argmax(passes.astype(jnp.int32), axis=1).astype(jnp.int32)
    raw_tags = order_tags[required_idx]
    del mac_ok
    dehnen_paper_mode = (
        jnp.asarray(policy_state.error_model_code, dtype=jnp.int32)
        == _ERROR_MODEL_DEHNEN_PAPER
    )
    accept_gate = jax.lax.cond(
        dehnen_paper_mode,
        lambda _: highest_order_pass,
        lambda _: highest_order_pass & allow_solver_override & relaxed_mac_ok,
        operand=None,
    )
    accept_mask = valid_pairs & different_nodes & accept_gate & pass_any
    tags = jnp.where(accept_mask, raw_tags, -jnp.ones_like(raw_tags))

    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=jnp.int32)
    actions = jnp.where(accept_mask, _ACTION_ACCEPT, actions)
    near_mask = (
        valid_pairs & different_nodes & target_leaf & source_leaf & (~accept_mask)
    )
    actions = jnp.where(near_mask, _ACTION_NEAR, actions)
    return actions, tags


def bucket_far_pairs_by_tag(
    interaction_sources: Array,
    interaction_targets: Array,
    interaction_tags: Array,
    num_tags: int,
) -> tuple[tuple[Array, Array], ...]:
    """Group accepted far pairs by integer tag."""

    buckets: list[tuple[Array, Array]] = []
    src = jnp.asarray(interaction_sources)
    tgt = jnp.asarray(interaction_targets)
    tags = jnp.asarray(interaction_tags)
    for idx in range(int(num_tags)):
        mask = tags == idx
        buckets.append((src[mask], tgt[mask]))
    return tuple(buckets)


__all__ = [
    "AdaptivePolicyState",
    "adaptive_pair_policy",
    "adaptive_policy_tolerance",
    "bucket_far_pairs_by_tag",
    "build_adaptive_policy_state",
    "compute_node_force_scale_from_sorted_acc",
    "compute_leaf_enclosing_sphere_geometry",
    "compute_leaf_ritter_sphere_geometry",
    "compute_smallest_enclosing_sphere_geometry",
    "compute_tree_merged_sphere_geometry",
    "merge_bounding_spheres",
    "resolve_dehnen_geometry",
    "dehnen_like_pair_error_by_order_from_degree_power",
    "dehnen_multipole_power_by_degree",
    "dehnen_paper_pair_error_by_order",
    "source_error_proxy_by_order_from_degree_power",
    "source_error_proxy_by_order_from_multipoles",
    "source_power_by_degree_from_multipoles",
]
