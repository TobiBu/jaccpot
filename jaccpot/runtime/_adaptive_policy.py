"""Solver-owned adaptive traversal policy helpers."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike
from yggdrax.tree import Tree

from jaccpot.upward.tree_expansions import TreeUpwardData

from .fmm_caches import _contains_tracer

_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2

_ERROR_MODEL_TAIL_PROXY = 0
_ERROR_MODEL_DEHNEN_DEGREE = 1
_ERROR_MODEL_DEHNEN_PAPER = 2


class AdaptivePolicyState(NamedTuple):
    """Solver-owned per-node summaries used by adaptive traversal policies.

    Attributes
    ----------
    source_error_proxy_by_order : Array
        Residual tail proxy per node and candidate order.
    source_degree_power : Array
        Per-node multipole power grouped by spherical-harmonic degree.
    source_dehnen_power : Array
        Dehnen's exact per-degree source power for eq (15).
    source_mass : Array
        Total mass of each source node, the ``M_A`` of eq (16a).
    source_mac_center : Array
        MAC centre of each source node.
    target_mac_center : Array
        MAC centre of each target node.
    source_radius_bound : Array
        Bounding radius of each source node about its MAC centre.
    target_radius_bound : Array
        Bounding radius of each target node about its MAC centre.
    target_accept_threshold : Array
        Per-target acceptance threshold, ``eps * min_b |a_b|`` in eq (16a).
    order_tags : Array
        Integer tag identifying the gear each accepted pair was assigned.
    order_values : Array
        Candidate expansion orders, as integers.
    order_values_float : Array
        The same candidate orders as floats, for the error arithmetic.
    dehnen_binomial_masked_by_order : Array
        Binomial factors of eq (15), masked per candidate order.
    dehnen_exponent_by_order : Array
        Exponents of eq (15) per candidate order.
    relaxed_theta_sq : Array
        Squared opening angle used by the geometric pre-filter.
    error_model_code : Array
        Which error model is active, encoded for the traced path.
    gravitational_constant : float
        Newton constant -- see the note on the field itself.
    mac_theta_max : float
        Geometric cap on the opening angle -- see the note on the field itself.
    """

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
    order_values_float: Array
    dehnen_binomial_masked_by_order: Array
    dehnen_exponent_by_order: Array
    relaxed_theta_sq: Array
    error_model_code: Array
    #: Newton constant. eq (16a) compares a *force* error ``G Etilde M_A / r^2``
    #: against ``eps * min_b |a_b|``, and the force scales on the right come from
    #: a G-scaled prepass -- so omitting G here runs the criterion at an effective
    #: tolerance of ``eps * G``. Defaults to 1 so existing hand-built states in
    #: the tests keep working unchanged.
    gravitational_constant: float = 1.0
    #: Geometric cap on the opening angle ``(rho_z + rho_s) / r`` for pairs the
    #: error test accepts. eq (16a) caps it only at 1.0, which is the *boundary of
    #: convergence* of the multipole series, not a safe operating point: at
    #: opening 0.99 a p=4 expansion has O(1) error, and eq (15)'s bound is derived
    #: under an assumption of convergence, so it under-predicts there. 1.0
    #: reproduces eq (16a) verbatim; a smaller value is a disclosed deviation.
    mac_theta_max: float = 1.0


def adaptive_policy_tolerance(
    *, theta: float, p_gears: tuple[int, ...], dtype: object
) -> Array:
    """Return a conservative solver-side adaptive tolerance derived from ``theta``.

    Parameters
    ----------
    theta : float
        Opening angle, or its squared form where the caller pre-squares it.
    p_gears : tuple[int, ...]
        Candidate expansion orders the adaptive policy may choose between.
    dtype : object
        Working dtype for the returned arrays.

    Returns
    -------
    Array
        A conservative solver-side tolerance, derived so the adaptive lane never accepts more than the scalar-theta lane would.

    Raises
    ------
    ValueError
        If the requested tolerance is not positive.
    """

    if len(p_gears) == 0:
        raise ValueError("adaptive policy tolerance requires non-empty p_gears")
    return jnp.asarray(float(theta) ** (max(int(v) for v in p_gears) + 2), dtype=dtype)


def _packed_total_order(multipole_packed: Array) -> int:
    packed = jnp.asarray(multipole_packed)
    return int(round(np.sqrt(int(packed.shape[1])) - 1))


def source_power_by_degree_from_multipoles(*, multipole_packed: Array) -> Array:
    """Return per-node multipole power grouped by spherical-harmonic degree.

    Parameters
    ----------
    multipole_packed : Array
        Packed multipole coefficients per node.

    Returns
    -------
    Array
        Per-node multipole power grouped by spherical-harmonic degree.
    """

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

    Dehnen (2014) equation (12) sums over the *complex* moments::

        P_n^2 = sum_{m=-n}^{n} (n-m)! (n+m)! |M_n^m|^2

    The complex-basis packing stores ``M_n^m`` directly at ``m = -n..n``, so the
    sum maps onto the slice as written. The real (Dehnen no-sqrt2) packing
    instead stores the pair ``(Re M_n^{|m|}, Im M_n^{|m|})`` at slots ``+|m|``
    and ``-|m|``, so ``|M_n^{|m|}|^2`` is split across two real slots -- and
    because the weight is symmetric under ``m -> -m``, each ``|m| != 0``
    magnitude contributes to eq (12) *twice*. Summing the real slots once
    therefore under-estimates ``P_n`` by up to a factor sqrt(2), which makes the
    acceptance criterion correspondingly too permissive and, worse, makes the
    estimator basis-dependent. Double the ``m != 0`` weights in the real basis.

    Invariant pinned by the unit tests: for a single point mass ``m`` at
    distance ``d`` the exact power is ``P_n = m * d**n`` for every degree,
    independent of direction and of the packed representation.

    Parameters
    ----------
    multipole_packed : Array
        Packed multipole coefficients per node.

    Returns
    -------
    Array
        Dehnen's exact per-degree source power, the input to eq (15).
    """

    packed = jnp.asarray(multipole_packed)
    total_p = _packed_total_order(packed)
    is_complex = jnp.iscomplexobj(packed)
    dtype = packed.real.dtype if is_complex else packed.dtype
    factorial = jnp.exp(jax.lax.lgamma(jnp.arange(2 * total_p + 1, dtype=dtype) + 1.0))
    powers: list[Array] = []
    for ell in range(total_p + 1):
        start = ell * ell
        stop = (ell + 1) * (ell + 1)
        degree_slice = packed[:, start:stop]
        m_vals = jnp.arange(-ell, ell + 1, dtype=jnp.int32)
        m_abs = jnp.abs(m_vals)
        weights = factorial[ell - m_abs] * factorial[ell + m_abs]
        if not is_complex:
            weights = weights * jnp.where(
                m_vals == 0,
                jnp.asarray(1.0, dtype=dtype),
                jnp.asarray(2.0, dtype=dtype),
            )
        weighted_sq = jnp.square(jnp.abs(degree_slice)) * weights[None, :]
        powers.append(jnp.sqrt(jnp.sum(weighted_sq, axis=1)))
    return jnp.stack(powers, axis=1)


def source_error_proxy_by_order_from_degree_power(
    *,
    degree_power: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Return the residual tail proxy for each candidate order from degree power.

    Parameters
    ----------
    degree_power : Array
        Per-node multipole power grouped by spherical-harmonic degree.
    p_gears : tuple[int, ...]
        Candidate expansion orders the adaptive policy may choose between.

    Returns
    -------
    Array
        The residual tail proxy for each candidate order.
    """

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
    """Return a Dehnen-style degree-weighted pair error estimate by order.

    Parameters
    ----------
    degree_power : Array
        Per-node multipole power grouped by spherical-harmonic degree.
    opening : Array
        Opening angle ``(rho_s + rho_t) / r``, clipped into ``[0, 1]``.
    order_values : Array
        Candidate expansion orders, as integers.

    Returns
    -------
    Array
        A degree-weighted pair error estimate per candidate order.

    Raises
    ------
    ValueError
        If the degree-power and order arrays disagree in shape.
    """

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
    order_values_float: Array,
    masked_binomial_by_order: Array,
    exponent_by_order: Array,
) -> Array:
    """Return Dehnen's equation (15) error estimate by candidate order.

    Parameters
    ----------
    source_power : Array
        Dehnen's per-degree source power for the source node.
    source_mass : Array
        Total source-node mass, the ``M_A`` of eq (16a).
    source_radius : Array
        Source-node bounding radius about its MAC centre.
    target_radius : Array
        Target-node bounding radius about its MAC centre.
    distance : Array
        Centre-to-centre separation of the pair.
    order_values_float : Array
        Candidate expansion orders as floats, for the error arithmetic.
    masked_binomial_by_order : Array
        Binomial factors of eq (15), masked per candidate order.
    exponent_by_order : Array
        Exponents of eq (15) per candidate order.

    Returns
    -------
    Array
        Dehnen's eq (15) error estimate per candidate order.
    """

    power = jnp.asarray(source_power)
    mass = jnp.asarray(source_mass, dtype=power.dtype)
    rho_z = jnp.asarray(source_radius, dtype=power.dtype)
    rho_s = jnp.asarray(target_radius, dtype=power.dtype)
    r = jnp.asarray(distance, dtype=power.dtype)
    order_values_arr = jnp.asarray(order_values_float, dtype=power.dtype)
    if int(order_values_arr.shape[0]) == 0:
        return jnp.zeros((power.shape[0], 0), dtype=power.dtype)
    tiny = jnp.asarray(1e-24, dtype=power.dtype)
    safe_mass = jnp.maximum(jnp.abs(mass), tiny)
    safe_r = jnp.maximum(r, tiny)
    rho_factor = jnp.power(
        rho_s[:, None, None],
        jnp.asarray(exponent_by_order, dtype=power.dtype)[None, :, :],
    )
    e_terms = (
        jnp.asarray(masked_binomial_by_order, dtype=power.dtype)[None, :, :]
        * power[:, None, :]
        * rho_factor
    )
    r_pow = jnp.power(safe_r[:, None], order_values_arr[None, :])
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
    """Compute a conservative per-node residual proxy for each candidate order.

    Parameters
    ----------
    multipole_packed : Array
        Packed multipole coefficients per node.
    p_gears : tuple[int, ...]
        Candidate expansion orders the adaptive policy may choose between.

    Returns
    -------
    Array
        A conservative per-node residual proxy per candidate order.
    """

    degree_power = source_power_by_degree_from_multipoles(
        multipole_packed=multipole_packed,
    )
    return source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=p_gears,
    )


def compute_node_force_scale_from_sorted_acc(
    *,
    tree: Tree,
    accelerations_sorted: Array,
    reduction: str = "max",
) -> Array:
    """Estimate per-node force scales from sorted per-particle accelerations.

    This is Dehnen eq (16a)'s right-hand side: the scale is ``|a_b|``, so the
    vector accelerations are reduced to magnitudes first. For eq (16b)'s
    cancellation-free ``f_b``, which is already a scalar per particle, call
    :func:`compute_node_force_scale_from_sorted_magnitudes` directly.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    accelerations_sorted : Array
        Per-particle accelerations in tree order.
    reduction : str
        How to reduce per-particle values onto a node: ``min``, ``mean`` or ``max``.

    Returns
    -------
    Array
        Per-node force scales reduced from the per-particle accelerations.
    """

    return compute_node_force_scale_from_sorted_magnitudes(
        tree=tree,
        magnitudes_sorted=jnp.linalg.norm(jnp.asarray(accelerations_sorted), axis=1),
        reduction=reduction,
    )


def compute_node_force_scale_from_sorted_magnitudes(
    *,
    tree: Tree,
    magnitudes_sorted: Array,
    reduction: str = "max",
) -> Array:
    """Reduce a sorted per-particle scalar scale onto every node of the tree.

    This is implemented as a JAX-native tree reduction: compute per-leaf scales
    from the contiguous particle blocks, then propagate scales upward through the
    binary tree using child reductions.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    magnitudes_sorted : Array
        Per-particle scalar magnitudes in tree order.
    reduction : str
        How to reduce per-particle values onto a node: ``min``, ``mean`` or ``max``.

    Returns
    -------
    Array
        Per-node scales reduced from a sorted per-particle scalar.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of the supported reductions.
    """

    reduction_norm = str(reduction).strip().lower()
    if reduction_norm not in ("max", "min"):
        raise ValueError("reduction must be 'max' or 'min'")
    use_min = reduction_norm == "min"

    magnitudes = jnp.asarray(magnitudes_sorted)
    if magnitudes.ndim != 1:
        raise ValueError(
            "magnitudes_sorted must be 1-D (one scalar scale per particle); "
            f"got shape {tuple(magnitudes.shape)}"
        )
    dtype = magnitudes.dtype
    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(tree.num_internal_nodes)
    if num_nodes == 0:
        return jnp.zeros((0,), dtype=dtype)

    identity = jnp.asarray(jnp.inf if use_min else -jnp.inf, dtype=dtype)
    scales = jnp.zeros((num_nodes,), dtype=dtype)
    leaf_ranges = node_ranges[num_internal:] if num_internal > 0 else node_ranges
    leaf_count = int(leaf_ranges.shape[0])

    if leaf_count > 0:
        counts = jnp.maximum(leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1, 0)
        max_leaf = int(jnp.max(counts)) if leaf_count > 0 else 0
        if max_leaf > 0:
            idx = jnp.arange(max_leaf, dtype=jnp.int32)
            particle_idx = leaf_ranges[:, 0:1] + idx[None, :]
            valid = idx[None, :] < counts[:, None]
            safe_idx = jnp.clip(particle_idx, 0, magnitudes.shape[0] - 1)
            leaf_values = magnitudes[safe_idx]
            leaf_values = jnp.where(valid, leaf_values, identity)
            leaf_scale = (
                jnp.min(leaf_values, axis=1)
                if use_min
                else jnp.max(leaf_values, axis=1)
            )
            # Empty leaves must take the reduction *identity*, not zero. Under the
            # `min` reduction used by the paper MAC, a zero here propagates all the
            # way to the root and drives `target_accept_threshold` to its floor, so
            # a single padded leaf refuses every far pair on its whole ancestor
            # chain. Capacity-fixed static_radix trees have padded leaves by
            # construction, so this is the common case, not a corner case.
            leaf_scale = jnp.where(
                counts > 0, leaf_scale, jnp.full_like(leaf_scale, identity)
            )
            leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=jnp.int32)
            scales = scales.at[leaf_nodes].set(leaf_scale)

    if num_internal <= 0:
        return scales

    left_child = jnp.asarray(tree.left_child, dtype=jnp.int32)
    right_child = jnp.asarray(tree.right_child, dtype=jnp.int32)
    zero = jnp.asarray(0.0, dtype=dtype)
    # Internal-node indices are not guaranteed to be stored in postorder.
    # Reduce in ascending node span so children are populated before parents.
    internal_ranges = node_ranges[:num_internal]
    internal_width = internal_ranges[:, 1] - internal_ranges[:, 0]
    internal_order = jnp.argsort(internal_width, stable=True)

    # Loop index: a `fori_loop` tracer, not an `int` -- see `body` below at the
    # Ritter sweep, which already annotates it `Array` (F40).
    def body(i: Array, current: Array) -> Array:
        node_idx = internal_order[i]
        left_idx = left_child[node_idx]
        right_idx = right_child[node_idx]
        left_valid = left_idx >= 0
        right_valid = right_idx >= 0
        left_value = jnp.where(left_valid, current[jnp.maximum(left_idx, 0)], identity)
        right_value = jnp.where(
            right_valid,
            current[jnp.maximum(right_idx, 0)],
            identity,
        )
        node_value = (
            jnp.minimum(left_value, right_value)
            if use_min
            else jnp.maximum(left_value, right_value)
        )
        node_value = jnp.where(left_valid | right_valid, node_value, zero)
        return current.at[node_idx].set(node_value)

    scales = jax.lax.fori_loop(0, num_internal, body, scales)
    # Nodes spanning no particles at all retain the reduction identity, which is
    # non-finite. They can never be a meaningful interaction target, but leaving
    # +/-inf in the array would poison any downstream arithmetic, so collapse them
    # onto the global finite extremum.
    finite = jnp.isfinite(scales)
    fallback = jnp.where(
        jnp.any(finite),
        (
            jnp.min(jnp.where(finite, scales, jnp.inf))
            if use_min
            else jnp.max(jnp.where(finite, scales, -jnp.inf))
        ),
        jnp.asarray(0.0, dtype=dtype),
    )
    return jnp.where(finite, scales, fallback)


def _sphere_from_support(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the exact sphere defined by up to four support points.

    Parameters
    ----------
    points : np.ndarray
        Point set the enclosing sphere is computed for, shape ``(k, 3)``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(centre, radius)`` of the sphere defined by the support set.

    Raises
    ------
    ValueError
        If the support set has more than four points.
    """

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
    """Return the exact smallest enclosing sphere for a 3D point set.

    Parameters
    ----------
    points : np.ndarray
        Point set the enclosing sphere is computed for, shape ``(k, 3)``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(centre, radius)`` of the exact smallest enclosing sphere.
    """

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
    """Return exact SES centres and radii for each node range.

    Parameters
    ----------
    node_ranges : Array
        Particle index range ``[lo, hi]`` per node.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.

    Returns
    -------
    tuple[Array, Array]
        ``(centres, radii)`` of the exact SES for every node.
    """

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
    """Return exact SES centres and radii for leaf nodes only.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.

    Returns
    -------
    tuple[Array, Array]
        ``(centres, radii)`` of the exact SES for the leaf nodes.
    """

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
    """Return approximate bounding spheres for padded leaf particle blocks.

    Parameters
    ----------
    leaf_points : Array
        Padded per-leaf point block, shape ``(num_leaves, cap, 3)``.
    leaf_valid : Array
        See the module docstring.

    Returns
    -------
    tuple[Array, Array]
        ``(centres, radii)`` from Ritter's approximation, batched over padded leaves.
    """

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

            # Loop index: a `fori_loop` tracer, not an `int` (F40).
            def body(i: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
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
    """Return fast approximate leaf spheres using a batched JAX Ritter pass.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.

    Returns
    -------
    tuple[Array, Array]
        ``(centres, radii)`` of the approximate leaf spheres.
    """

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


def compute_center_referenced_radius_geometry(
    *, tree: Tree, positions_sorted: Array, centers: Array
) -> Array:
    """Return per-node radii measured about ``centers``, not about a fitted sphere.

    Dehnen's error estimate (eqs 13/15) needs ``rho`` measured about the point
    the expansion is actually taken about. Feeding it a *smallest-enclosing-
    sphere* radius while the multipoles are expanded about the centre of mass
    under-bounds the true radius (the SES radius is minimal by construction) and
    over-accepts. This helper measures the radius about whatever centre the
    runtime actually uses, so the MAC distance and the M2L displacement are the
    same quantity.

    Exact at leaves; a valid upper bound at internal nodes, via the standard
    ``|c_child - c_parent| + rho_child`` merge. Fully device-side and
    differentiable, unlike the Welzl/Ritter sphere fits.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.
    centers : Array
        Reference centre per node.

    Returns
    -------
    Array
        Per-node radii measured about the supplied centres.
    """

    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    pos = jnp.asarray(positions_sorted)
    node_centers = jnp.asarray(centers, dtype=pos.dtype)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(tree.num_internal_nodes)
    dtype = pos.dtype
    radii = jnp.zeros((num_nodes,), dtype=dtype)
    if num_nodes == 0:
        return radii

    leaf_ranges = node_ranges[num_internal:] if num_internal > 0 else node_ranges
    if int(leaf_ranges.shape[0]) > 0:
        counts = jnp.maximum(leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1, 0)
        max_leaf = int(jnp.max(counts))
        if max_leaf > 0:
            idx = jnp.arange(max_leaf, dtype=jnp.int32)
            particle_idx = leaf_ranges[:, 0:1] + idx[None, :]
            valid = idx[None, :] < counts[:, None]
            safe_idx = jnp.clip(particle_idx, 0, pos.shape[0] - 1)
            leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=jnp.int32)
            offsets = pos[safe_idx] - node_centers[leaf_nodes][:, None, :]
            reach = jnp.linalg.norm(offsets, axis=2)
            reach = jnp.where(valid, reach, jnp.asarray(0.0, dtype=dtype))
            leaf_radii = jnp.max(reach, axis=1)
            leaf_radii = jnp.where(counts > 0, leaf_radii, jnp.zeros_like(leaf_radii))
            radii = radii.at[leaf_nodes].set(leaf_radii)

    if num_internal <= 0:
        return radii

    left_child = jnp.asarray(tree.left_child, dtype=jnp.int32)
    right_child = jnp.asarray(tree.right_child, dtype=jnp.int32)
    # Ascending node span so children are populated before parents; radix
    # internal-node indices are not stored in postorder.
    internal_ranges = node_ranges[:num_internal]
    internal_order = jnp.argsort(
        internal_ranges[:, 1] - internal_ranges[:, 0], stable=True
    )

    def body(iter_idx: Array, state: Array) -> Array:
        node_idx = internal_order[iter_idx]
        center = node_centers[node_idx]

        def child_reach(child_idx: Array) -> Array:
            safe = jnp.maximum(child_idx, 0)
            offset = jnp.linalg.norm(node_centers[safe] - center)
            return jnp.where(
                child_idx >= 0,
                offset + state[safe],
                jnp.asarray(0.0, dtype=dtype),
            )

        merged = jnp.maximum(
            child_reach(left_child[node_idx]), child_reach(right_child[node_idx])
        )
        return state.at[node_idx].set(merged)

    return jax.lax.fori_loop(0, num_internal, body, radii)


def merge_bounding_spheres(
    center_a: Array, radius_a: Array, center_b: Array, radius_b: Array
) -> tuple[Array, Array]:
    """Return the minimal sphere containing two spheres.

    Parameters
    ----------
    center_a : Array
        See the module docstring.
    radius_a : Array
        See the module docstring.
    center_b : Array
        See the module docstring.
    radius_b : Array
        See the module docstring.

    Returns
    -------
    tuple[Array, Array]
        ``(centre, radius)`` of the minimal sphere containing both inputs.
    """

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
    """Return node spheres from leaf spheres and JAX upward merges.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.
    leaf_mode : str
        See the module docstring.

    Returns
    -------
    tuple[Array, Array]
        ``(centres, radii)`` for every node, merged upward from the leaves.

    Raises
    ------
    ValueError
        If the leaf spheres and tree shape disagree.
    """

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
    # Internal-node indices are not guaranteed to be stored in postorder, so a
    # descending-index sweep can visit a parent before its children and merge
    # spheres that are still unpopulated -- the resulting node sphere then fails
    # to contain its own particles, which silently breaks the MAC's `theta < 1`
    # convergence guard. Merge in ascending node span instead, matching
    # `compute_node_force_scale_from_sorted_acc`.
    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    internal_ranges = node_ranges[:num_internal]
    internal_width = internal_ranges[:, 1] - internal_ranges[:, 0]
    internal_order = jnp.argsort(internal_width, stable=True)

    def body(iter_idx: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        center_state, radius_state = state
        node_idx = internal_order[iter_idx]
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


#: Geometry modes whose leaf pass is a numpy host loop and therefore cannot be
#: traced. ``exact`` runs an exact smallest-enclosing-sphere solve per node;
#: ``tree`` runs the same solve per leaf before the device-side merge.
_HOST_ONLY_DEHNEN_GEOMETRY_MODES = ("exact", "tree")

#: Smallest per-node opening angle the effective-theta mode will emit. A node
#: demanding a smaller opening than this needs an extent large enough to dominate
#: the pair queue; in practice such nodes are better refined than accepted.
_EFFECTIVE_THETA_FLOOR = 1e-3

#: Radius floor, as a fraction of the root radius, applied before the extent
#: rescale. Not cosmetic: ``_propagate_extents`` replaces any extent ``<= 0`` with
#: an ancestor's, and ``_compute_leaf_effective_extents`` replaces zero-extent
#: *leaves* with a root-derived depth padding -- either path silently discards the
#: per-node scale. A single-particle leaf has radius exactly 0, so this is the
#: common case, not a corner case.
_EFFECTIVE_THETA_RADIUS_FLOOR_FRAC = 1e-9


def per_node_effective_theta(
    *,
    source_power: Array,
    radius_bound: Array,
    force_scale: Array,
    masked_binomial: Array,
    exponent: Array,
    order: int,
    eps: float,
    gravitational_constant: float = 1.0,
    theta_floor: float = _EFFECTIVE_THETA_FLOOR,
    theta_max: float = 1.0,
) -> Array:
    """Collapse Dehnen eq (16a)/(16b) into one opening angle per node.

    The exact criterion needs a solver-owned ``pair_policy``, and that vetoes every
    production fast lane. This turns it into a *per-node* opening angle so
    acceptance becomes the plain scalar-theta comparison the lanes already run; see
    :func:`per_node_mac_radius` for how the two are made algebraically identical.

    **The inversion is closed form.** In eq (15) the distance enters only as
    ``r**(p+2)``, and ``improvement`` depends solely on the two radii, so for fixed
    radii the eq (16a) left-hand side is exactly ``C_i / r**(p+2)``. Verified
    numerically: ``est_force_error * (r/rho)**(p+2)`` is constant to full double
    precision across a 6x range of ``r``. Hence

        theta_i = (rho_z + rho_s) * (eps * s_i / C_i) ** (1 / (p + 2))
        C_i     = improvement * G * sum_n C(p, n) * P_n^(i) * rho_s ** (p - n)

    with ``s_i`` the node's force scale -- ``min_b |a_b|`` for eq (16a) or
    ``min_b f_b`` for eq (16b), which is just a different array here.

    No root-find is required, so nothing leaves the device: this is one fused
    expression over concrete arrays. An earlier plan called for ~5 Newton/bisection
    steps per node *on the host*; that was predicated on the relation being
    non-invertible, which it is not.

    ``P_n`` is used **as measured**. The cruder ``P_n/M <= rho**n`` bound collapses
    the sum to ``M(2 rho)**p`` and costs the entire multipole spectrum -- and since
    the criterion's measured advantage is concentrated in the error *tail*, the
    spectrum is exactly what must not be discarded.

    **The approximation that remains, and it is not small.** eq (16a) pairs the
    *source's* mass and power against the *sink's* force scale. A per-node angle
    cannot express that: yggdrax's ``mac_extents`` is a single array indexed by both
    ``safe_targets`` and ``safe_sources``, so a node's extent is identical in either
    role. This function therefore pairs each node's own power with its own force
    scale, under the self-similar assumption ``rho_z = rho_s = rho_i`` (whence
    ``improvement == 4`` and ``rho_z + rho_s == 2 rho_i``).

    That conflation is *not* guaranteed conservative. A massive node sitting where
    accelerations are high gets a permissive ``theta_i``, yet as a source acting on
    a distant low-acceleration sink the exact criterion would demand far more
    separation. Those are precisely the large-dynamic-range configurations the
    criterion exists to exploit, and where it wins biggest (bulge+halo p99 5.9-85x
    against Plummer's 1.3-2.2x). **Accept-mask agreement and matched-work tail error
    against the exact criterion must be measured before this mode is used for
    anything quantitative**, bulge+halo especially.

    Parameters
    ----------
    source_power : Array
        Dehnen per-degree power ``P_n``, shape ``(num_nodes, total_p + 1)``.
    radius_bound : Array
        Per-node radius about the expansion centre, shape ``(num_nodes,)``.
    force_scale : Array
        Per-node force scale on the criterion's right-hand side, ``(num_nodes,)``.
    masked_binomial : Array
        ``C(p, n)`` for ``n <= p`` else 0, for the chosen order: ``(total_p + 1,)``.
    exponent : Array
        ``max(p - n, 0)`` for the chosen order, shape ``(total_p + 1,)``.
    order : int
        Expansion order ``p`` these weights correspond to.
    eps : float
        Relative force-accuracy target of eq (16a).
    gravitational_constant : float
        ``G``. eq (16a) compares a force error against ``eps * min_b |a_b|`` and the
        force scale is G-scaled, so omitting it runs at an effective ``eps * G``.
    theta_floor : float
        Lower clamp on the returned angle.
    theta_max : float
        Upper clamp. At 1.0 this keeps the paper's own convergence guard.

    Returns
    -------
    Array
        Per-node effective opening angle, shape ``(num_nodes,)``.
    """

    power = jnp.asarray(source_power)
    dtype = power.dtype
    rho = jnp.asarray(radius_bound, dtype=dtype)
    scale = jnp.asarray(force_scale, dtype=dtype)
    weights = jnp.asarray(masked_binomial, dtype=dtype)
    powers = jnp.asarray(exponent, dtype=dtype)
    tiny = jnp.asarray(1e-300, dtype=dtype)
    p = int(order)

    # Self-similar pairing: rho_z = rho_s = rho, so `improvement` collapses to the
    # constant 4 and the pair separation is 2 * rho.
    improvement = jnp.asarray(4.0, dtype=dtype)
    rho_safe = jnp.maximum(rho, tiny)
    aggregate = jnp.sum(
        weights[None, :] * power * jnp.power(rho_safe[:, None], powers[None, :]),
        axis=1,
    )
    coefficient = (
        improvement * jnp.asarray(gravitational_constant, dtype=dtype) * aggregate
    )

    # A node whose power vanishes above the monopole is a point mass: C -> 0 sends
    # theta -> inf and the clamp takes it to theta_max, which is correct -- its
    # expansion is exact at any opening.
    ratio = (jnp.asarray(eps, dtype=dtype) * jnp.maximum(scale, tiny)) / jnp.maximum(
        coefficient, tiny
    )
    theta = (
        jnp.asarray(2.0, dtype=dtype)
        * rho_safe
        * jnp.power(ratio, jnp.asarray(1.0 / (p + 2.0), dtype=dtype))
    )
    theta = jnp.where(jnp.isfinite(theta), theta, jnp.asarray(theta_max, dtype=dtype))
    return jnp.clip(
        theta,
        jnp.asarray(theta_floor, dtype=dtype),
        jnp.asarray(theta_max, dtype=dtype),
    )


def per_node_conservative_extent(
    *,
    source_mass: Array,
    radius_bound: Array,
    force_scale: Array,
    order: int,
    eps: float,
    gravitational_constant: float = 1.0,
    geometric_gain: float = 2.0,
    split_lambda: Optional[float] = None,
) -> tuple[Array, float]:
    """Per-node MAC extents that provably never over-accept relative to eq (16a).

    Motivation: :func:`per_node_effective_theta` matches eq (16a) exactly for a
    self-similar pair but pairs each node's own power with its own force scale,
    which measured 12-9300x worse than the exact criterion at up to 15x more work.
    This instead builds a *sound* bound -- if the traversal accepts, eq (16a) would
    have accepted too. It will accept strictly less than the exact criterion; the
    question it exists to answer is whether it still beats fixed-theta.

    **Derivation.** eq (16a) for source A on sink B is satisfied when

        8 G M_A (rho_A + rho_B)**p / (eps s_B) <= r**(p+2)

    (using the crude source bound ``P_n <= M rho**n``, which collapses the eq (15)
    sum to ``M (rho_A + rho_B)**p``, and ``improvement <= 8``). Enforcing
    ``e_i >= c rho_i`` makes ``e_A + e_B <= r`` imply ``rho_A + rho_B <= r / c``,
    so ``(rho_A+rho_B)**p <= r**p / c**p`` and the requirement reduces to

        r**2 >= 8 G M_A / (eps s_B c**p)

    The remaining ``M_A / s_B`` is a *product* of a source quantity and a sink
    quantity, and the lane test is a *sum*, so it cannot be represented exactly --
    this is the same obstruction that sinks the effective-theta mode. Split it with
    AM-GM, ``sqrt(x y) <= (x/lam + lam y)/2``:

        e_i = max( c rho_i,  K M_i / lam,  K lam / s_i ),
        K   = sqrt(8 G / (eps c**p)) / 2

    Taking the max over both roles makes one array serve as source and sink.

    **Where this is loose, by construction.** AM-GM is tight only when
    ``lam**2 == M_A s_B``. A single global ``lam`` must therefore absorb the whole
    dynamic range of mass and force scale across the tree -- and that range is
    precisely what the mass-dependent criterion exploits. Expect the bound to be
    weakest on the distributions where the criterion is most valuable (bulge+halo).

    ``geometric_gain`` (``c``) trades the two terms: raising it shrinks the mass
    term by ``c**(p/2)`` while growing the geometric term linearly, so at p=8 a
    modest ``c=2`` buys a 16x reduction for a 2x cost. There is an optimum; the
    default is a starting point, not a tuned value.

    Parameters
    ----------
    source_mass : Array
        Per-node spanned mass, shape ``(num_nodes,)``.
    radius_bound : Array
        Per-node radius about the expansion centre, shape ``(num_nodes,)``.
    force_scale : Array
        Per-node force scale on the criterion's right-hand side, ``(num_nodes,)``.
    order : int
        Expansion order ``p``.
    eps : float
        Relative force-accuracy target of eq (16a).
    gravitational_constant : float
        ``G``. The force scale is G-scaled, so omitting it runs at ``eps * G``.
    geometric_gain : float
        ``c`` in the split above: grows the geometric term linearly while shrinking
        the mass term by ``c**(p/2)``. The default is a starting point, not tuned.
    split_lambda : Optional[float]
        Fraction of the error budget given to the mass term; ``None`` chooses it.

    Returns
    -------
    tuple[Array, float]
        Per-node extents, and the ``split_lambda`` actually used (handy when it was
        chosen automatically).
    """

    mass = jnp.asarray(source_mass)
    dtype = mass.dtype
    rho = jnp.asarray(radius_bound, dtype=dtype)
    scale = jnp.asarray(force_scale, dtype=dtype)
    tiny = jnp.asarray(1e-300, dtype=dtype)
    p = int(order)
    c = float(geometric_gain)

    scale_safe = jnp.maximum(scale, tiny)
    if split_lambda is None:
        # Equalise the two AM-GM terms at the geometric median of the pair
        # distribution: lam**2 ~ typical(M) * typical(s).
        typical_mass = jnp.exp(jnp.mean(jnp.log(jnp.maximum(mass, tiny))))
        typical_scale = jnp.exp(jnp.mean(jnp.log(scale_safe)))
        lam_arr = jnp.sqrt(typical_mass * typical_scale)
        lam = float(lam_arr)
    else:
        lam = float(split_lambda)
    lam_safe = max(lam, float(jnp.finfo(dtype).tiny))

    coefficient = jnp.sqrt(
        jnp.asarray(
            8.0 * float(gravitational_constant) / (float(eps) * c**p), dtype=dtype
        )
    ) / jnp.asarray(2.0, dtype=dtype)
    source_term = coefficient * mass / jnp.asarray(lam_safe, dtype=dtype)
    sink_term = coefficient * jnp.asarray(lam_safe, dtype=dtype) / scale_safe
    geometric_term = jnp.asarray(c, dtype=dtype) * rho
    extents = jnp.maximum(geometric_term, jnp.maximum(source_term, sink_term))
    return extents, lam_safe


def per_node_mac_radius(
    *,
    radius_bound: Array,
    theta_nodes: Array,
    theta_global: float,
    radius_floor_frac: float = _EFFECTIVE_THETA_RADIUS_FLOOR_FRAC,
) -> Array:
    """Rescale node radii so a scalar-theta lane test reproduces per-node angles.

    The lanes compare ``(e_t + e_s)**2 <= theta_g**2 * d**2``. Feeding
    ``e_i = rho_i * theta_g / theta_i`` makes that *algebraically identical* to

        rho_t / theta_t + rho_s / theta_s <= d

    so ``theta_g`` cancels out of acceptance entirely -- the desired property, since
    it matches paper mode where the global theta does not gate. The result is a
    per-node rescale of ``geometry.radius``, not a new comparison, so no
    ``pair_policy`` is needed and none of the lane vetoes trigger.

    The radius floor is load-bearing, not defensive -- see
    ``_EFFECTIVE_THETA_RADIUS_FLOOR_FRAC``.

    Parameters
    ----------
    radius_bound : Array
        See the module docstring.
    theta_nodes : Array
        See the module docstring.
    theta_global : float
        See the module docstring.
    radius_floor_frac : float
        See the module docstring.

    Returns
    -------
    Array
        Node radii rescaled so a scalar-theta acceptance test reproduces the per-node criterion.
    """

    rho = jnp.asarray(radius_bound)
    dtype = rho.dtype
    theta = jnp.asarray(theta_nodes, dtype=dtype)
    root_radius = jnp.max(rho)
    floor = jnp.maximum(
        jnp.asarray(radius_floor_frac, dtype=dtype) * root_radius,
        jnp.asarray(float(jnp.finfo(dtype).tiny), dtype=dtype),
    )
    rho_floored = jnp.maximum(rho, floor)
    return rho_floored * (jnp.asarray(theta_global, dtype=dtype) / theta)


def resolve_dehnen_geometry(
    *,
    geometry_mode: Literal["com", "exact", "tree", "tree_approx", "runtime"],
    tree: Tree,
    positions_sorted: Array,
    upward: TreeUpwardData,
    dtype: DTypeLike,
) -> tuple[Array, Array]:
    """Return MAC centres and radii for the requested Dehnen geometry mode.

    ``com`` (the default) references both the centres and the radii to the
    runtime's own expansion centres, so the distance entering eqs (13)/(15) is
    exactly the M2L displacement. The remaining modes fit a separate bounding
    sphere per node and are kept as opt-in reference modes for comparing
    Dehnen's section 5.1 centre recommendations; because their radii are
    measured about the fitted sphere centre rather than the expansion centre,
    they are inflated by the centre offset to stay valid bounds.

    Parameters
    ----------
    geometry_mode : Literal['com', 'exact', 'tree', 'tree_approx', 'runtime']
        Which MAC geometry to build: com, aabb or an enclosing sphere.
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.
    upward : TreeUpwardData
        Upward-sweep artifacts supplying the multipoles and geometry.
    dtype : DTypeLike
        Working dtype for the returned arrays.

    Returns
    -------
    tuple[Array, Array]
        ``(mac_centres, mac_radii)`` for the requested geometry mode.

    Raises
    ------
    RuntimeError
        If the requested geometry could not be built for this tree.
    ValueError
        If ``geometry_mode`` is not a supported value.
    """

    mode = str(geometry_mode).strip().lower()
    if mode in _HOST_ONLY_DEHNEN_GEOMETRY_MODES and _contains_tracer(
        (positions_sorted, tree.node_ranges)
    ):
        raise RuntimeError(
            f"dehnen_geometry_mode={mode!r} runs a numpy host loop over nodes and "
            "cannot be traced; use 'com' (recommended), 'tree_approx', or "
            "'runtime' under jax.jit/jax.grad"
        )
    if mode == "com":
        mac_centers = jnp.asarray(upward.multipoles.centers, dtype=dtype)
        # Two independent valid bounds on the radius about the expansion centre:
        #   (1) leaf-exact, merged upward as |c_child - c_i| + rho_child. Tight
        #       near the leaves but accumulates slack with depth.
        #   (2) the exact farthest-corner distance from the centre to the node's
        #       axis-aligned bounding box. Depth-independent, but loose whenever
        #       the box is emptier than it is large.
        # Neither dominates, so take the elementwise minimum -- still a bound,
        # and tighter than either alone.
        merged_bound = compute_center_referenced_radius_geometry(
            tree=tree,
            positions_sorted=positions_sorted,
            centers=mac_centers,
        )
        geometry_centers = jnp.asarray(upward.geometry.center, dtype=dtype)
        half_extent = jnp.asarray(upward.geometry.half_extent, dtype=dtype)
        aabb_bound = jnp.linalg.norm(
            jnp.abs(mac_centers - geometry_centers) + half_extent, axis=1
        )
        radius_bound = jnp.minimum(jnp.asarray(merged_bound, dtype=dtype), aabb_bound)
        return mac_centers, radius_bound
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
        mac_centers = jnp.asarray(upward.geometry.center, dtype=dtype)
        radius_bound = jnp.asarray(upward.geometry.radius, dtype=dtype)
    else:
        raise ValueError(
            "dehnen_geometry_mode must be 'com', 'exact', 'tree', 'tree_approx', "
            "or 'runtime'"
        )
    # Every sphere-fit mode measures its radius about the *fitted* centre, but
    # the multipoles are expanded -- and the M2L translation applied -- about
    # `upward.multipoles.centers`. Re-reference to the expansion centre and
    # inflate by the offset so the pair distance entering eqs (13)/(15) is the
    # true M2L displacement and the radius stays a valid bound about it.
    expansion_centers = jnp.asarray(upward.multipoles.centers, dtype=dtype)
    fitted_centers = jnp.asarray(mac_centers, dtype=dtype)
    center_offset = jnp.linalg.norm(expansion_centers - fitted_centers, axis=1)
    radius_bound = jnp.asarray(radius_bound, dtype=dtype) + center_offset
    return expansion_centers, radius_bound


def _dehnen_binomial_matrix(
    *, p_gears: tuple[int, ...], total_p: int, dtype: DTypeLike
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
    gravitational_constant: float = 1.0,
    mac_theta_max: float = 1.0,
) -> AdaptivePolicyState:
    """Build the solver-owned adaptive traversal state from upward data.

    Parameters
    ----------
    upward : TreeUpwardData
        Upward-sweep artifacts supplying the multipoles and geometry.
    tree : Tree
        The tree whose nodes are being summarised.
    positions_sorted : Array
        Particle positions in tree order, shape ``(N, 3)``.
    p_gears : tuple[int, ...]
        Candidate expansion orders the adaptive policy may choose between.
    force_scale_nodes : Optional[Array]
        Per-node force scale from the prepass.
    eps : Array
        Relative force-accuracy target of eq (16a).
    theta : Array
        Opening angle, or its squared form where the caller pre-squares it.
    error_model_code : Array
        See the module docstring.
    dehnen_geometry_mode : str
        See the module docstring.
    gravitational_constant : float
        See the module docstring.
    mac_theta_max : float
        See the module docstring.

    Returns
    -------
    AdaptivePolicyState
        The solver-owned per-node summaries the traversal policy reads.

    Raises
    ------
    ValueError
        If the inputs are mutually inconsistent, or a required target is missing.
    """

    if len(p_gears) == 0:
        raise ValueError("adaptive policy state requires non-empty p_gears")
    packed = upward.multipoles.packed
    dtype = packed.real.dtype if jnp.iscomplexobj(packed) else packed.dtype
    num_nodes = int(packed.shape[0])
    error_model_code_arr = jnp.asarray(error_model_code, dtype=jnp.int32)
    model_code = int(error_model_code_arr)

    total_p = _packed_total_order(packed)
    empty_degree = jnp.zeros((num_nodes, total_p + 1), dtype=dtype)
    empty_order = jnp.zeros((num_nodes, len(p_gears)), dtype=dtype)
    degree_power = empty_degree
    dehnen_power = empty_degree
    error_proxy = empty_order

    if model_code == _ERROR_MODEL_TAIL_PROXY:
        degree_power = source_power_by_degree_from_multipoles(multipole_packed=packed)
        error_proxy = source_error_proxy_by_order_from_degree_power(
            degree_power=degree_power,
            p_gears=p_gears,
        )
    elif model_code == _ERROR_MODEL_DEHNEN_DEGREE:
        degree_power = source_power_by_degree_from_multipoles(multipole_packed=packed)
    elif model_code == _ERROR_MODEL_DEHNEN_PAPER:
        dehnen_power = dehnen_multipole_power_by_degree(multipole_packed=packed)
        total_p = int(dehnen_power.shape[1] - 1)
    else:
        raise ValueError(f"unknown adaptive error model code {model_code}")

    exact_dehnen = model_code == _ERROR_MODEL_DEHNEN_PAPER
    if force_scale_nodes is None:
        target_force_scale = jnp.ones((num_nodes,), dtype=dtype)
    else:
        target_force_scale = jnp.asarray(force_scale_nodes, dtype=dtype)
        if int(target_force_scale.shape[0]) != num_nodes:
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
    order_values_float = order_values.astype(dtype)
    theta_arr = jnp.asarray(theta, dtype=dtype)
    relaxed_theta = jnp.minimum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(1.5, dtype=dtype) * theta_arr,
    )
    if exact_dehnen:
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
    if exact_dehnen:
        degree_idx = jnp.arange(total_p + 1, dtype=jnp.int32)
        exponent_by_order = jnp.maximum(order_values[:, None] - degree_idx[None, :], 0)
        masked_binomial_by_order = _dehnen_binomial_matrix(
            p_gears=p_gears,
            total_p=total_p,
            dtype=dtype,
        ) * (degree_idx[None, :] <= order_values[:, None]).astype(dtype)
    else:
        exponent_by_order = jnp.zeros((len(p_gears), total_p + 1), dtype=jnp.int32)
        masked_binomial_by_order = jnp.zeros((len(p_gears), total_p + 1), dtype=dtype)
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
        order_values_float=order_values_float,
        dehnen_binomial_masked_by_order=masked_binomial_by_order,
        dehnen_exponent_by_order=exponent_by_order,
        relaxed_theta_sq=jnp.square(relaxed_theta),
        error_model_code=error_model_code_arr,
        gravitational_constant=float(gravitational_constant),
        mac_theta_max=float(mac_theta_max),
    )


def adaptive_pair_policy(
    policy_state: AdaptivePolicyState, **pair_data: Array
) -> tuple[Array, Array]:
    """Return traversal actions and order tags from solver-owned adaptive state.

    Parameters
    ----------
    policy_state : AdaptivePolicyState
        The solver-owned adaptive policy state.
    **pair_data : Array
        The traversal's per-pair arrays. Keyword-splatted because yggdrax owns the
        call and may add fields, but the keys this policy actually reads are a
        contract: ``valid_pairs``, ``mac_ok``, ``different_nodes``, ``target_leaf``,
        ``source_leaf``, ``target_nodes``, ``source_nodes``, ``dist_sq``,
        ``extent_target`` and ``extent_source``. A missing key is a ``KeyError`` at
        trace time, not a silent default.

    Returns
    -------
    tuple[Array, Array]
        ``(actions, order_tags)``: the traversal action per pair and the gear each
        accepted pair was assigned.
    """

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
    target_threshold = jnp.asarray(policy_state.target_accept_threshold)[safe_targets]

    def _tail_proxy(_: None) -> Array:
        source_proxy = jnp.asarray(policy_state.source_error_proxy_by_order)[
            safe_sources, :
        ]
        return (
            jnp.square(source_proxy) * extent_sum_sq[:, None]
            < jnp.square(target_threshold)[:, None] * safe_dist_sq[:, None]
        )

    def _dehnen_degree(_: None) -> Array:
        source_degree_power = jnp.asarray(policy_state.source_degree_power)[
            safe_sources, :
        ]
        opening = jnp.sqrt(extent_sum_sq / safe_dist_sq)
        pair_error = dehnen_like_pair_error_by_order_from_degree_power(
            degree_power=source_degree_power,
            opening=opening,
            order_values=policy_state.order_values,
        )
        return pair_error < target_threshold[:, None]

    def _dehnen_paper_directional(src: Array, tgt: Array) -> Array:
        """Evaluate eq (16a) for sources ``src`` acting on targets ``tgt``.

        Parameters
        ----------
        src : Array
            Source node indices.
        tgt : Array
            Target node indices.

        Returns
        -------
        Array
            Eq (16a) evaluated for the given source acting on the given target.
        """

        source_dehnen_power = jnp.asarray(policy_state.source_dehnen_power)[src, :]
        source_mass = jnp.asarray(policy_state.source_mass)[src]
        source_radius = jnp.asarray(policy_state.source_radius_bound)[src]
        target_radius = jnp.asarray(policy_state.target_radius_bound)[tgt]
        source_mac_center = jnp.asarray(policy_state.source_mac_center)[src]
        target_mac_center = jnp.asarray(policy_state.target_mac_center)[tgt]
        threshold = jnp.asarray(policy_state.target_accept_threshold)[tgt]
        paper_distance = jnp.maximum(
            jnp.linalg.norm(source_mac_center - target_mac_center, axis=1),
            jnp.asarray(1e-24, dtype=dist_sq.dtype),
        )
        pair_error = dehnen_paper_pair_error_by_order(
            source_power=source_dehnen_power,
            source_mass=source_mass,
            source_radius=source_radius,
            target_radius=target_radius,
            distance=paper_distance,
            order_values_float=policy_state.order_values_float,
            masked_binomial_by_order=policy_state.dehnen_binomial_masked_by_order,
            exponent_by_order=policy_state.dehnen_exponent_by_order,
        )
        # eq (16a) left-hand side: the estimated *force* error contributed by this
        # single interaction, G Etilde M_A / r^2.
        est_force_error = (
            pair_error
            * jnp.asarray(policy_state.gravitational_constant, dtype=pair_error.dtype)
            * source_mass[:, None]
            / jnp.maximum(
                jnp.square(paper_distance[:, None]),
                jnp.asarray(1e-24, dtype=pair_error.dtype),
            )
        )
        # eq (16a) first clause, generalised: `theta < mac_theta_max` with
        # mac_theta_max = 1 giving the paper's own `theta < 1`.
        convergent = (source_radius + target_radius) < (
            jnp.asarray(policy_state.mac_theta_max, dtype=paper_distance.dtype)
            * paper_distance
        )
        return convergent[:, None] & (est_force_error < threshold[:, None])

    def _dehnen_paper(_: None) -> Array:
        # eq (16a) is genuinely asymmetric in A<->B: it uses the *source* mass and
        # multipole power against the *sink's* own force scale. The traversal,
        # however, evaluates this policy in both orientations and only accepts on
        # `accept_both` / marks near on `near_both`; a leaf-leaf pair whose two
        # orientations disagree therefore falls through to REFINE, and a leaf-leaf
        # pair cannot be refined -- so it is dropped entirely, receiving neither an
        # M2L nor a P2P contribution. Paper mode is uniquely exposed because it is
        # the one model that does not gate acceptance on `allow_solver_override`.
        # Symmetrize here so both orientations return the same decision; the
        # effective tolerance is then the stricter of the two directions.
        return _dehnen_paper_directional(
            safe_sources, safe_targets
        ) & _dehnen_paper_directional(safe_targets, safe_sources)

    passes = jax.lax.switch(
        jnp.asarray(policy_state.error_model_code, dtype=jnp.int32),
        (_tail_proxy, _dehnen_degree, _dehnen_paper),
        operand=None,
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
    """Group accepted far pairs by integer tag.

    Parameters
    ----------
    interaction_sources : Array
        See the module docstring.
    interaction_targets : Array
        See the module docstring.
    interaction_tags : Array
        See the module docstring.
    num_tags : int
        Number of distinct tags to bucket into.

    Returns
    -------
    tuple[tuple[Array, Array], ...]
        One ``(sources, targets)`` pair of arrays per tag, in tag order.
    """

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
    "compute_node_force_scale_from_sorted_magnitudes",
    "per_node_effective_theta",
    "per_node_mac_radius",
    "compute_center_referenced_radius_geometry",
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
