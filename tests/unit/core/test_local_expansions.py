# flake8: noqa
# ruff: noqa: E203
"""Tests for local expansion helpers."""

import itertools

import jax
import jax.numpy as jnp
import pytest

from yggdrax.dense_interactions import build_dense_interactions
from jaccpot.runtime.fmm import FastMultipoleMethod
from jaccpot.downward.local_expansions import (
    DEFAULT_M2L_CHUNK_SIZE,
    LocalExpansionData,
    TreeDownwardData,
    accumulate_dense_m2l_contributions,
    accumulate_m2l_contributions,
    initialize_local_expansions,
    prepare_downward_sweep,
    propagate_local_expansions,
    run_downward_sweep,
    translate_local_expansion,
    translate_multipole_to_local,
)
from jaccpot.operators.multipole_utils import (
    level_offset,
    multi_index_factorial,
    multi_index_tuples,
    total_coefficients,
)
from yggdrax.tree import build_tree
from jaccpot.upward.tree_expansions import prepare_upward_sweep
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import build_well_separated_interactions
from yggdrax.tree_moments import multipole_from_packed

DEFAULT_TEST_LEAF_SIZE = 1


def _build_tree_and_centers():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.1, 0.0, 0.3],
            [0.6, 0.4, -0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    geom = compute_tree_geometry(tree, pos_sorted)
    return tree, geom.center


def _prepare_tree_for_m2l(order: int = 2):
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    upward = prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
    )
    interactions = build_well_separated_interactions(
        tree,
        upward.geometry,
        theta=2.0,
    )
    return tree, upward, interactions


def _pack_local_coefficients(order, scalar, vec, mat, dtype):
    coeffs = jnp.zeros((total_coefficients(order),), dtype=dtype)
    coeffs = coeffs.at[level_offset(0)].set(scalar)

    if order >= 1:
        packed_vec = jnp.stack([vec[2], vec[1], vec[0]])
        offset1 = level_offset(1)
        coeffs = coeffs.at[offset1 : offset1 + 3].set(packed_vec)

    if order >= 2:
        packed_mat = jnp.stack(
            [
                mat[2, 2],
                mat[1, 2],
                mat[1, 1],
                mat[0, 2],
                mat[0, 1],
                mat[0, 0],
            ]
        )
        offset2 = level_offset(2)
        coeffs = coeffs.at[offset2 : offset2 + 6].set(packed_mat)

    return coeffs


def _evaluate_local(coeffs, point, order):
    result = coeffs[level_offset(0)]

    if order >= 1:
        dip_packed = coeffs[level_offset(1) : level_offset(1) + 3]
        dip_vec = jnp.stack([dip_packed[2], dip_packed[1], dip_packed[0]])
        result = result + jnp.dot(dip_vec, point)

    if order >= 2:
        quad_packed = coeffs[level_offset(2) : level_offset(2) + 6]
        quad_mat = jnp.stack(
            [
                jnp.stack([quad_packed[5], quad_packed[4], quad_packed[3]]),
                jnp.stack([quad_packed[4], quad_packed[2], quad_packed[1]]),
                jnp.stack([quad_packed[3], quad_packed[1], quad_packed[0]]),
            ]
        )
        result = result + 0.5 * point @ quad_mat @ point

    return result


def _unpack_local_coefficients(coeffs, order):
    scalar = coeffs[level_offset(0)]

    vec = jnp.zeros((3,), dtype=coeffs.dtype)
    if order >= 1:
        packed = coeffs[level_offset(1) : level_offset(1) + 3]
        vec = jnp.stack([packed[2], packed[1], packed[0]])

    mat = jnp.zeros((3, 3), dtype=coeffs.dtype)
    if order >= 2:
        packed = coeffs[level_offset(2) : level_offset(2) + 6]
        mat = jnp.stack(
            [
                jnp.stack([packed[5], packed[4], packed[3]]),
                jnp.stack([packed[4], packed[2], packed[1]]),
                jnp.stack([packed[3], packed[1], packed[0]]),
            ]
        )

    return scalar, vec, mat


def _multi_power(
    point: jnp.ndarray,
    combo: tuple[int, int, int],
) -> jnp.ndarray:
    value = jnp.array(1.0, dtype=point.dtype)
    if combo[0]:
        value = value * point[0] ** combo[0]
    if combo[1]:
        value = value * point[1] ** combo[1]
    if combo[2]:
        value = value * point[2] ** combo[2]
    return value


def _pack_polynomial_derivatives(
    order: int,
    coeff_map: dict[tuple[int, int, int], jnp.ndarray],
    dtype,
) -> jnp.ndarray:
    total = total_coefficients(order)
    dtype = jnp.dtype(dtype)
    coeffs = jnp.zeros((total,), dtype=dtype)
    for level in range(order + 1):
        start = level_offset(level)
        combos = multi_index_tuples(level)
        values = []
        for combo in combos:
            factor = multi_index_factorial(combo)
            scaled = coeff_map[combo] * dtype.type(factor)
            values.append(jnp.asarray(scaled, dtype=dtype))
        coeffs = coeffs.at[start : start + len(combos)].set(
            jnp.asarray(values, dtype=dtype)
        )
    return coeffs


def _evaluate_local_series(
    coeffs: jnp.ndarray, point: jnp.ndarray, order: int
) -> jnp.ndarray:
    dtype = coeffs.dtype
    total = dtype.type(0.0)
    for level in range(order + 1):
        start = level_offset(level)
        combos = multi_index_tuples(level)
        coeff_slice = coeffs[start : start + len(combos)]
        for idx, combo in enumerate(combos):
            factor = dtype.type(1.0 / multi_index_factorial(combo))
            term = coeff_slice[idx] * _multi_power(point, combo) * factor
            total = total + term
    return total


def _pack_multipole_from_expansion(
    expansion,
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    order: int,
    dtype,
) -> jnp.ndarray:
    total = total_coefficients(order)
    dtype = jnp.dtype(dtype)
    coeffs = jnp.zeros((total,), dtype=dtype)
    coeffs = coeffs.at[level_offset(0)].set(
        jnp.asarray(expansion.monopole, dtype=dtype)
    )

    rel = jnp.asarray(positions, dtype=dtype) - jnp.asarray(
        expansion.center,
        dtype=dtype,
    )
    mass_vec = jnp.asarray(masses, dtype=dtype)

    dipole = jnp.einsum("n,ni->i", mass_vec, rel)
    second = jnp.einsum("n,ni,nj->ij", mass_vec, rel, rel)
    third = jnp.einsum("n,ni,nj,nk->ijk", mass_vec, rel, rel, rel)
    fourth = jnp.einsum(
        "n,ni,nj,nk,nl->ijkl",
        mass_vec,
        rel,
        rel,
        rel,
        rel,
    )

    tensors = {
        1: dipole,
        2: second,
        3: third,
        4: fourth,
    }

    for level in range(1, order + 1):
        combos = multi_index_tuples(level)
        tensor = tensors[level]
        values = []
        for combo in combos:
            index = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
            values.append(tensor[index])
        coeffs = coeffs.at[level_offset(level) : level_offset(level) + len(combos)].set(
            jnp.asarray(values, dtype=dtype)
        )

    return coeffs


def _unpack_local_derivatives(coeffs: jnp.ndarray, order: int):
    derivatives = []
    dtype = coeffs.dtype
    for level in range(order + 1):
        start = level_offset(level)
        combos = multi_index_tuples(level)
        slice_vals = coeffs[start : start + len(combos)]
        if level == 0:
            derivatives.append(slice_vals[0])
            continue
        tensor = jnp.zeros((3,) * level, dtype=dtype)
        for idx, combo in enumerate(combos):
            indices = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
            value = slice_vals[idx]
            for perm in set(itertools.permutations(indices)):
                tensor = tensor.at[perm].set(value)
        derivatives.append(tensor)
    return derivatives


def _compute_multipole_moments(positions, masses, center):
    rel = positions - center
    mass = jnp.sum(masses)
    dipole = jnp.sum(masses[:, None] * rel, axis=0)
    second = jnp.einsum("n,ni,nj->ij", masses, rel, rel)
    trace = jnp.trace(second)
    identity = jnp.eye(3, dtype=positions.dtype)
    quadrupole = 3.0 * second - trace * identity
    return mass, dipole, quadrupole


def _multipole_potential(r_vec, mass, dipole, quadrupole, order):
    inv_r = jnp.reciprocal(jnp.sqrt(jnp.dot(r_vec, r_vec)))
    result = mass * inv_r

    if order >= 1:
        result = result + jnp.dot(dipole, r_vec) * inv_r**3

    if order >= 2:
        quad_term = jnp.dot(r_vec, quadrupole @ r_vec)
        result = result + 0.5 * quad_term * inv_r**5

    return result


def test_initialize_local_expansions_shapes():
    tree, centers = _build_tree_and_centers()
    data = initialize_local_expansions(tree, centers, max_order=2)

    assert isinstance(data, LocalExpansionData)
    assert data.order == 2

    num_nodes = tree.parent.shape[0]
    expected_coeffs = total_coefficients(2)
    assert data.coefficients.shape == (num_nodes, expected_coeffs)
    assert jnp.allclose(data.coefficients, 0.0)
    assert jnp.allclose(data.centers, centers)


def test_initialize_local_expansions_requires_matching_centers():
    tree, centers = _build_tree_and_centers()
    bad_centers = jnp.zeros((centers.shape[0] - 1, 3))

    with pytest.raises(ValueError):
        initialize_local_expansions(tree, bad_centers, max_order=1)

    with pytest.raises(ValueError):
        initialize_local_expansions(tree, centers, max_order=-1)


def test_translate_local_expansion_matches_quadratic_shift():
    order = 2
    dtype = jnp.float64

    scalar = dtype(1.7)
    vec = jnp.array([0.3, -0.2, 0.5], dtype=dtype)
    mat = jnp.array(
        [
            [0.9, -0.4, 0.2],
            [-0.4, 0.7, -0.1],
            [0.2, -0.1, -0.5],
        ],
        dtype=dtype,
    )

    coeffs = _pack_local_coefficients(order, scalar, vec, mat, dtype)
    delta = jnp.array([0.25, -0.35, 0.15], dtype=dtype)

    shifted = translate_local_expansion(coeffs, delta, order=order)

    test_points = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [-0.05, 0.2, -0.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    for point in test_points:
        parent_eval = _evaluate_local(coeffs, point + delta, order)
        child_eval = _evaluate_local(shifted, point, order)
        assert jnp.allclose(parent_eval, child_eval)


def test_translate_local_expansion_handles_order_four_polynomial():
    order = 4
    dtype = jnp.dtype(jnp.float64)

    combos = [
        combo for level in range(order + 1) for combo in multi_index_tuples(level)
    ]
    key = jax.random.PRNGKey(13)
    coeff_values = jax.random.normal(key, (len(combos),), dtype=dtype)
    coeff_map = {combo: coeff_values[idx] for idx, combo in enumerate(combos)}

    coeffs = _pack_polynomial_derivatives(order, coeff_map, dtype)
    delta = jnp.array([0.18, -0.12, 0.09], dtype=dtype)
    translated = translate_local_expansion(coeffs, delta, order=order)

    rho_points = jnp.array(
        [
            [0.02, -0.03, 0.01],
            [-0.015, 0.02, -0.025],
            [0.03, 0.01, -0.02],
        ],
        dtype=dtype,
    )

    def evaluate_parent(point):
        return _evaluate_local_series(coeffs, point, order)

    def evaluate_child(point):
        return _evaluate_local_series(translated, point, order)

    for idx in range(rho_points.shape[0]):
        rho = rho_points[idx]
        parent_val = evaluate_parent(rho + delta)
        child_val = evaluate_child(rho)
        assert jnp.allclose(child_val, parent_val, atol=1e-11, rtol=1e-9)

        grad_parent = jax.grad(evaluate_parent)(rho + delta)
        grad_child = jax.grad(evaluate_child)(rho)
        assert jnp.allclose(grad_child, grad_parent, atol=1e-10, rtol=1e-8)


def test_translate_multipole_to_local_matches_direct_derivatives():
    """M2L translation reproduces the truncated multipole derivatives."""

    positions = jnp.array(
        [
            [0.35, -0.1, 0.2],
            [-0.4, 0.5, -0.15],
            [0.1, -0.3, 0.45],
            [-0.2, -0.25, -0.35],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.2, 0.9, 1.5, 0.8], dtype=jnp.float64)

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
        return_reordered=True,
    )

    max_order = 4
    upward = prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=max_order,
    )

    multipoles = upward.multipoles
    root_index = 0
    center_source = multipoles.centers[root_index]
    center_target = jnp.array([1.8, -0.6, 0.9], dtype=jnp.float64)
    delta = center_target - center_source

    coefficients = translate_multipole_to_local(
        multipoles.packed[root_index],
        delta,
        order=max_order,
        raw_mass=multipoles.moments.mass[root_index],
        raw_dipole=multipoles.moments.dipole[root_index],
        raw_second=multipoles.moments.second_moment[root_index],
        raw_third=multipoles.moments.third_moment[root_index],
        raw_fourth=multipoles.moments.fourth_moment[root_index],
    )

    truncated = multipole_from_packed(
        multipoles.packed[root_index][jnp.newaxis, :],
        center_source[jnp.newaxis, :],
        max_order,
    )

    mass = truncated.mass[0]
    dipole = truncated.dipole[0]
    quadrupole = truncated.quadrupole[0]
    octupole = truncated.octupole[0]
    hexadecapole = truncated.hexadecapole[0]

    def multipole_potential(displacement):
        inv_r = jnp.reciprocal(jnp.linalg.norm(displacement))
        phi_val = mass * inv_r
        if max_order >= 1:
            phi_val = phi_val + jnp.dot(dipole, displacement) * inv_r**3
        if max_order >= 2:
            quad_term = jnp.einsum(
                "ij,i,j->",
                quadrupole,
                displacement,
                displacement,
            )
            phi_val = phi_val + 0.5 * quad_term * inv_r**5
        if max_order >= 3:
            oct_term = jnp.einsum(
                "ijk,i,j,k->",
                octupole,
                displacement,
                displacement,
                displacement,
            )
            phi_val = phi_val + (1.0 / 6.0) * oct_term * inv_r**7
        if max_order >= 4:
            hex_term = jnp.einsum(
                "ijkl,i,j,k,l->",
                hexadecapole,
                displacement,
                displacement,
                displacement,
                displacement,
            )
            phi_val = phi_val + (1.0 / 24.0) * hex_term * inv_r**9
        return phi_val

    def phi(offset):
        return multipole_potential(delta + offset)

    def _take_axis_derivative(func, axis):
        gradient = jax.grad(func)

        def component(x):
            return gradient(x)[axis]

        return component

    expected = []
    zero = jnp.zeros((3,), dtype=jnp.float64)
    for level in range(max_order + 1):
        for combo in multi_index_tuples(level):
            derived = phi
            for axis, count in enumerate(combo):
                for _ in range(count):
                    derived = _take_axis_derivative(derived, axis)
            expected.append(derived(zero))

    expected_arr = jnp.array(expected, dtype=jnp.float64)

    assert jnp.allclose(
        coefficients[: expected_arr.shape[0]],
        expected_arr,
        rtol=5e-9,
        atol=5e-9,
    )


def test_propagate_local_expansions_accumulates_parent():
    tree, centers = _build_tree_and_centers()
    order = 2
    data = initialize_local_expansions(tree, centers, max_order=order)

    scalar = centers.dtype.type(0.8)
    vec = jnp.array([0.05, -0.03, 0.04], dtype=centers.dtype)
    mat = jnp.array(
        [
            [0.2, 0.01, -0.02],
            [0.01, -0.15, 0.03],
            [-0.02, 0.03, 0.12],
        ],
        dtype=centers.dtype,
    )
    root_coeffs = _pack_local_coefficients(
        order,
        scalar,
        vec,
        mat,
        centers.dtype,
    )

    coeffs = data.coefficients.at[0].set(root_coeffs)

    leaf_index = int(tree.num_internal_nodes)
    leaf_scalar = centers.dtype.type(-0.6)
    leaf_vec = jnp.array([0.02, 0.01, -0.04], dtype=centers.dtype)
    leaf_mat = jnp.array(
        [
            [0.05, -0.02, 0.01],
            [-0.02, 0.07, -0.03],
            [0.01, -0.03, 0.04],
        ],
        dtype=centers.dtype,
    )
    leaf_coeffs = _pack_local_coefficients(
        order,
        leaf_scalar,
        leaf_vec,
        leaf_mat,
        centers.dtype,
    )
    coeffs = coeffs.at[leaf_index].set(leaf_coeffs)

    local_data = LocalExpansionData(order, centers, coeffs)
    propagated = propagate_local_expansions(tree, local_data)

    delta_leaf = centers[leaf_index] - centers[0]
    translated_root = translate_local_expansion(
        root_coeffs,
        delta_leaf,
        order=order,
    )
    expected_leaf = leaf_coeffs + translated_root

    assert jnp.allclose(propagated.coefficients[leaf_index], expected_leaf)


@pytest.mark.parametrize(
    "chunk_size",
    [1, 2, DEFAULT_M2L_CHUNK_SIZE, 10**6],
)
def test_accumulate_m2l_matches_pairwise_translations(chunk_size):
    tree, upward, interactions = _prepare_tree_for_m2l()

    order = int(upward.multipoles.order)
    local_init = initialize_local_expansions(
        tree,
        upward.multipoles.centers,
        max_order=order,
    )
    accumulated = accumulate_m2l_contributions(
        interactions,
        upward.multipoles,
        local_init,
        chunk_size=chunk_size,
    )

    centers_target = jnp.asarray(local_init.centers)
    centers_source = jnp.asarray(upward.multipoles.centers)

    total_nodes = tree.parent.shape[0]
    saw_interaction = False
    for target in range(total_nodes):
        start = int(interactions.offsets[target])
        count = int(interactions.counts[target])
        end = start + count
        coeffs_target = accumulated.coefficients[target]
        if start == end:
            assert jnp.allclose(
                coeffs_target,
                jnp.zeros_like(coeffs_target),
            )
            continue

        saw_interaction = True
        expected = jnp.zeros_like(coeffs_target)
        for idx in range(start, end):
            source = int(interactions.sources[idx])
            delta = centers_target[target] - centers_source[source]
            translated = translate_multipole_to_local(
                upward.multipoles.packed[source],
                delta,
                order=order,
            )
            expected = expected + translated

        assert jnp.allclose(coeffs_target, expected)

    assert saw_interaction


@pytest.mark.parametrize("bad_chunk", [0, -3])
def test_accumulate_m2l_rejects_non_positive_chunk_size(bad_chunk):
    tree, upward, interactions = _prepare_tree_for_m2l()

    order = int(upward.multipoles.order)
    local_init = initialize_local_expansions(
        tree,
        upward.multipoles.centers,
        max_order=order,
    )

    with pytest.raises(ValueError):
        accumulate_m2l_contributions(
            interactions,
            upward.multipoles,
            local_init,
            chunk_size=bad_chunk,
        )


def test_accumulate_dense_m2l_matches_sparse_accumulator():
    tree, upward, _interactions = _prepare_tree_for_m2l()
    dense_buffers = build_dense_interactions(
        tree,
        upward.geometry,
        theta=2.0,
    )
    local_init = initialize_local_expansions(
        tree,
        upward.multipoles.centers,
        max_order=int(upward.multipoles.order),
    )

    dense_locals = accumulate_dense_m2l_contributions(
        dense_buffers,
        upward.multipoles,
        local_init,
    )
    sparse_locals = accumulate_m2l_contributions(
        dense_buffers.sparse_interactions,
        upward.multipoles,
        local_init,
    )

    assert jnp.allclose(dense_locals.coefficients, sparse_locals.coefficients)


def test_translate_multipole_to_local_matches_autodiff_reference():
    order = 2
    dtype = jnp.float64

    positions = jnp.array(
        [
            [0.15, -0.05, 0.08],
            [-0.12, 0.09, -0.04],
            [0.05, 0.04, -0.11],
        ],
        dtype=dtype,
    )
    masses = jnp.array([1.5, 0.7, 1.2], dtype=dtype)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=dtype),
    )

    tree, positions_sorted, masses_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    upward = prepare_upward_sweep(
        tree,
        positions_sorted,
        masses_sorted,
        max_order=order,
    )

    center = upward.multipoles.centers[0]
    multipole_coeffs = upward.multipoles.packed[0]

    delta = jnp.array([1.8, -1.3, 0.9], dtype=dtype)
    local_coeffs = translate_multipole_to_local(
        multipole_coeffs,
        delta,
        order=order,
    )

    scalar, vec, mat = _unpack_local_coefficients(local_coeffs, order)

    moments = multipole_from_packed(
        multipole_coeffs[jnp.newaxis, :],
        center[jnp.newaxis, :],
        order,
    )
    mass = moments.mass[0]
    dipole = moments.dipole[0]
    quadrupole = moments.quadrupole[0]

    def potential(displacement):
        return _multipole_potential(
            displacement,
            mass,
            dipole,
            quadrupole,
            order,
        )

    phi_ref = potential(delta)
    grad_ref = jax.grad(potential)(delta)
    hess_ref = jax.hessian(potential)(delta)

    assert jnp.allclose(scalar, phi_ref)
    assert jnp.allclose(vec, grad_ref)
    assert jnp.allclose(mat, hess_ref)

    rho = jnp.array([2.0e-4, -1.5e-4, 1.2e-4], dtype=dtype)
    approx = _evaluate_local(local_coeffs, rho, order)
    actual = potential(delta + rho)
    assert jnp.allclose(approx, actual, atol=1e-10, rtol=1e-8)


def test_translate_multipole_to_local_order_four_matches_autodiff():
    order = 4
    dtype = jnp.dtype(jnp.float64)

    positions = jnp.array(
        [
            [0.35, -0.27, 0.18],
            [-0.22, 0.31, -0.29],
            [0.12, 0.14, -0.33],
            [-0.28, -0.19, 0.26],
        ],
        dtype=dtype,
    )
    masses = jnp.array([1.1, 0.9, 1.3, 0.8], dtype=dtype)

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=dtype),
    )

    tree, positions_sorted, masses_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    upward = prepare_upward_sweep(
        tree,
        positions_sorted,
        masses_sorted,
        max_order=order,
    )

    center = upward.multipoles.centers[0]
    multipole_coeffs = upward.multipoles.packed[0]
    delta = jnp.array([0.4, -0.35, 0.3], dtype=dtype)
    local_coeffs = translate_multipole_to_local(
        multipole_coeffs,
        delta,
        order=order,
    )

    scalar, gradient, hessian, third, fourth = _unpack_local_derivatives(
        local_coeffs,
        order,
    )[:5]

    moments = multipole_from_packed(
        multipole_coeffs[jnp.newaxis, :],
        center[jnp.newaxis, :],
        order,
    )
    mass = moments.mass[0]
    dipole = moments.dipole[0]
    quadrupole = moments.quadrupole[0]
    octupole = moments.octupole[0]
    hexadecapole = moments.hexadecapole[0]

    def multipole_potential(displacement):
        inv_r = jnp.reciprocal(jnp.linalg.norm(displacement))
        phi_val = mass * inv_r
        if order >= 1:
            phi_val = phi_val + jnp.dot(dipole, displacement) * inv_r**3
        if order >= 2:
            quad_term = jnp.einsum(
                "ij,i,j->",
                quadrupole,
                displacement,
                displacement,
            )
            phi_val = phi_val + 0.5 * quad_term * inv_r**5
        if order >= 3:
            oct_term = jnp.einsum(
                "ijk,i,j,k->",
                octupole,
                displacement,
                displacement,
                displacement,
            )
            phi_val = phi_val + (1.0 / 6.0) * oct_term * inv_r**7
        if order >= 4:
            hex_term = jnp.einsum(
                "ijkl,i,j,k,l->",
                hexadecapole,
                displacement,
                displacement,
                displacement,
                displacement,
            )
            phi_val = phi_val + (1.0 / 24.0) * hex_term * inv_r**9
        return phi_val

    grad_fn = jax.grad(multipole_potential)
    hess_fn = jax.jacfwd(grad_fn)
    third_fn = jax.jacfwd(hess_fn)
    fourth_fn = jax.jacfwd(third_fn)

    phi_ref = multipole_potential(delta)
    grad_ref = grad_fn(delta)
    hess_ref = hess_fn(delta)
    third_ref = third_fn(delta)
    fourth_ref = fourth_fn(delta)

    assert jnp.allclose(scalar, phi_ref, atol=1e-10, rtol=1e-8)
    assert jnp.allclose(gradient, grad_ref, atol=1e-10, rtol=1e-8)
    assert jnp.allclose(hessian, hess_ref, atol=1e-9, rtol=1e-7)
    assert jnp.allclose(third, third_ref, atol=1e-9, rtol=1e-7)
    assert jnp.allclose(fourth, fourth_ref, atol=1e-8, rtol=1e-6)

    offsets = jnp.array(
        [
            [0.01, -0.015, 0.012],
            [-0.008, 0.01, -0.006],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    for idx in range(offsets.shape[0]):
        rho = offsets[idx]
        approx = _evaluate_local_series(local_coeffs, rho, order)
        actual = multipole_potential(delta + rho)
    assert jnp.allclose(approx, actual, atol=1e-8, rtol=5e-6)


def test_translate_multipole_to_local_order4_derivatives_offsets():
    order = 4
    dtype = jnp.dtype(jnp.float64)

    positions = jnp.array(
        [
            [0.35, -0.27, 0.18],
            [-0.22, 0.31, -0.29],
            [0.12, 0.14, -0.33],
            [-0.28, -0.19, 0.26],
        ],
        dtype=dtype,
    )
    masses = jnp.array([1.1, 0.9, 1.3, 0.8], dtype=dtype)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=dtype),
    )

    jax.config.update("jax_enable_x64", True)

    tree, positions_sorted, masses_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    upward = prepare_upward_sweep(
        tree,
        positions_sorted,
        masses_sorted,
        max_order=order,
    )

    multipole_coeffs = upward.multipoles.packed[0]
    center = upward.multipoles.centers[0]

    moments = multipole_from_packed(
        multipole_coeffs[jnp.newaxis, :],
        center[jnp.newaxis, :],
        order,
    )

    mass = moments.mass[0]
    dipole = moments.dipole[0]
    quadrupole = moments.quadrupole[0]
    octupole = moments.octupole[0]
    hexadecapole = moments.hexadecapole[0]

    def multipole_potential(displacement):
        inv_r = jnp.reciprocal(jnp.linalg.norm(displacement))
        value = mass * inv_r
        value = value + jnp.dot(dipole, displacement) * inv_r**3
        quad_term = jnp.einsum(
            "ij,i,j->",
            quadrupole,
            displacement,
            displacement,
        )
        value = value + 0.5 * quad_term * inv_r**5
        oct_term = jnp.einsum(
            "ijk,i,j,k->",
            octupole,
            displacement,
            displacement,
            displacement,
        )
        value = value + (1.0 / 6.0) * oct_term * inv_r**7
        hex_term = jnp.einsum(
            "ijkl,i,j,k,l->",
            hexadecapole,
            displacement,
            displacement,
            displacement,
            displacement,
        )
        value = value + (1.0 / 24.0) * hex_term * inv_r**9
        return value

    grad_fn = jax.grad(multipole_potential)
    hess_fn = jax.jacfwd(grad_fn)
    third_fn = jax.jacfwd(hess_fn)
    fourth_fn = jax.jacfwd(third_fn)

    offsets = jnp.array(
        [
            [0.4, -0.35, 0.3],
            [-0.45, 0.32, -0.28],
            [0.25, 0.41, -0.37],
        ],
        dtype=dtype,
    )

    for idx in range(offsets.shape[0]):
        delta = offsets[idx]
        local_coeffs = translate_multipole_to_local(
            multipole_coeffs,
            delta,
            order=order,
        )
        derivatives = _unpack_local_derivatives(local_coeffs, order)
        scalar = derivatives[0]
        gradient = derivatives[1]
        hessian = derivatives[2]
        third = derivatives[3]
        fourth = derivatives[4]

        phi_ref = multipole_potential(delta)
        grad_ref = grad_fn(delta)
        hess_ref = hess_fn(delta)
        third_ref = third_fn(delta)
        fourth_ref = fourth_fn(delta)

        assert jnp.allclose(scalar, phi_ref, atol=1e-10, rtol=1e-8)
        assert jnp.allclose(gradient, grad_ref, atol=1e-10, rtol=1e-8)
        assert jnp.allclose(hessian, hess_ref, atol=1e-9, rtol=1e-7)
        assert jnp.allclose(third, third_ref, atol=1e-9, rtol=1e-7)
        assert jnp.allclose(fourth, fourth_ref, atol=1e-8, rtol=1e-6)


def test_translate_multipole_to_local_far_field_matches_direct_sum():
    order = 4
    dtype = jnp.dtype(jnp.float64)

    key = jax.random.PRNGKey(123)
    src_positions = 0.15 * jax.random.normal(key, (6, 3), dtype=dtype)
    src_masses = jnp.abs(jax.random.normal(key, (6,), dtype=dtype)) + dtype.type(0.2)

    fmm = FastMultipoleMethod(G=1.0)
    expansion = fmm.compute_expansion(src_positions, src_masses, order=order)
    packed = _pack_multipole_from_expansion(
        expansion,
        src_positions,
        src_masses,
        order,
        dtype,
    )

    # Increase separation so order-4 truncation error falls below the strict
    # tolerance used in this test.
    separation_scale = dtype.type(24.0)
    delta = jnp.array([2.7, -1.9, 3.25], dtype=dtype) * separation_scale
    local_coeffs = translate_multipole_to_local(
        packed,
        delta,
        order=order,
    )

    key, subkey = jax.random.split(key)
    offsets = 0.05 * jax.random.normal(subkey, (5, 3), dtype=dtype)

    target_center = expansion.center + delta
    target_points = target_center + offsets

    def potential(point):
        diff = point - src_positions
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=1) + dtype.type(1e-20))
        return jnp.sum(src_masses / dist)

    local_potential = jax.vmap(
        lambda rho: _evaluate_local_series(local_coeffs, rho, order)
    )
    direct_potential = jax.vmap(potential)

    local_phi = local_potential(offsets)
    direct_phi = direct_potential(target_points)

    assert jnp.allclose(local_phi, direct_phi, atol=1e-10, rtol=1e-8)

    local_gradient_fn = jax.grad(
        lambda rho: _evaluate_local_series(local_coeffs, rho, order)
    )
    local_grad = jax.vmap(local_gradient_fn)(offsets)

    def direct_gradient(point):
        diff = point - src_positions
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=1) + dtype.type(1e-20))
        inv_dist3 = jnp.reciprocal(dist**3)
        return -jnp.sum(
            src_masses[:, None] * inv_dist3[:, None] * diff,
            axis=0,
        )

    direct_grad = jax.vmap(direct_gradient)(target_points)

    assert jnp.allclose(local_grad, direct_grad, atol=1e-10, rtol=1e-8)


def test_run_downward_sweep_matches_manual_sequence():
    tree, upward, interactions = _prepare_tree_for_m2l()

    order = int(upward.multipoles.order)
    base_locals = initialize_local_expansions(
        tree,
        upward.multipoles.centers,
        max_order=order,
    )

    dtype = base_locals.centers.dtype
    root_scalar = dtype.type(0.42)
    root_vec = jnp.array([0.01, -0.03, 0.02], dtype=dtype)
    root_mat = jnp.array(
        [
            [0.05, -0.01, 0.0],
            [-0.01, 0.07, 0.02],
            [0.0, 0.02, -0.04],
        ],
        dtype=dtype,
    )
    root_coeffs = _pack_local_coefficients(
        order,
        root_scalar,
        root_vec,
        root_mat,
        dtype,
    )

    coeffs_with_root = base_locals.coefficients.at[0].set(root_coeffs)
    initial_locals = LocalExpansionData(
        order,
        base_locals.centers,
        coeffs_with_root,
    )

    manual = propagate_local_expansions(
        tree,
        accumulate_m2l_contributions(
            interactions,
            upward.multipoles,
            initial_locals,
        ),
    )

    combined = run_downward_sweep(
        tree,
        upward.multipoles,
        interactions,
        initial_locals=initial_locals,
    )

    assert jnp.allclose(combined.coefficients, manual.coefficients)
    assert jnp.allclose(combined.centers, manual.centers)


def test_run_downward_sweep_handles_dense_buffers():
    tree, upward, interactions = _prepare_tree_for_m2l()
    dense_buffers = build_dense_interactions(
        tree,
        upward.geometry,
        theta=2.0,
    )

    sparse = run_downward_sweep(
        tree,
        upward.multipoles,
        interactions,
    )
    dense = run_downward_sweep(
        tree,
        upward.multipoles,
        dense_buffers.sparse_interactions,
        dense_buffers=dense_buffers,
    )

    assert jnp.allclose(dense.coefficients, sparse.coefficients)
    assert jnp.allclose(dense.centers, sparse.centers)


def test_run_downward_sweep_order_four_matches_manual_sequence():
    tree, upward, interactions = _prepare_tree_for_m2l(order=4)

    order = int(upward.multipoles.order)
    base_locals = initialize_local_expansions(
        tree,
        upward.multipoles.centers,
        max_order=order,
    )

    dtype = base_locals.centers.dtype
    root_scalar = dtype.type(0.18)
    root_vec = jnp.array([0.02, -0.015, 0.03], dtype=dtype)
    root_mat = jnp.array(
        [
            [0.04, -0.01, 0.0],
            [-0.01, 0.05, 0.015],
            [0.0, 0.015, -0.035],
        ],
        dtype=dtype,
    )
    root_coeffs = _pack_local_coefficients(
        order,
        root_scalar,
        root_vec,
        root_mat,
        dtype,
    )

    coeffs_with_root = base_locals.coefficients.at[0].set(root_coeffs)
    initial_locals = LocalExpansionData(
        order,
        base_locals.centers,
        coeffs_with_root,
    )

    manual = propagate_local_expansions(
        tree,
        accumulate_m2l_contributions(
            interactions,
            upward.multipoles,
            initial_locals,
        ),
    )

    combined = run_downward_sweep(
        tree,
        upward.multipoles,
        interactions,
        initial_locals=initial_locals,
    )

    assert jnp.allclose(combined.coefficients, manual.coefficients)
    assert jnp.allclose(combined.centers, manual.centers)


def test_prepare_downward_sweep_builds_expected_data():
    tree, upward, interactions = _prepare_tree_for_m2l()

    downward = prepare_downward_sweep(
        tree,
        upward,
        theta=2.0,
    )

    assert isinstance(downward, TreeDownwardData)
    assert jnp.array_equal(
        downward.interactions.offsets,
        interactions.offsets,
    )
    assert jnp.array_equal(
        downward.interactions.sources,
        interactions.sources,
    )
    assert jnp.array_equal(
        downward.interactions.level_offsets,
        interactions.level_offsets,
    )
    assert jnp.array_equal(
        downward.interactions.target_levels,
        interactions.target_levels,
    )

    manual = run_downward_sweep(
        tree,
        upward.multipoles,
        interactions,
    )
    assert jnp.allclose(downward.locals.coefficients, manual.coefficients)
    assert jnp.allclose(downward.locals.centers, manual.centers)


def test_prepare_downward_sweep_accepts_dense_buffers():
    tree, upward, _interactions = _prepare_tree_for_m2l()
    dense_buffers = build_dense_interactions(
        tree,
        upward.geometry,
        theta=2.0,
    )

    dense_downward = prepare_downward_sweep(
        tree,
        upward,
        theta=2.0,
        dense_buffers=dense_buffers,
    )
    sparse_downward = prepare_downward_sweep(
        tree,
        upward,
        theta=2.0,
    )

    assert jnp.allclose(
        dense_downward.locals.coefficients,
        sparse_downward.locals.coefficients,
    )
    assert jnp.allclose(
        dense_downward.locals.centers,
        sparse_downward.locals.centers,
    )
