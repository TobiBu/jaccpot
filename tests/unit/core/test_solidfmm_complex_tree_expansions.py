"""Tests for solidfmm complex upward sweep helpers."""

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import build_tree

from jaccpot.runtime._fmm_impl import FastMultipoleMethod
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.7, -0.4, -0.1],
            [-0.2, 0.1, 0.5],
            [0.4, -0.3, 0.2],
            [0.8, 0.6, -0.4],
            [0.1, 0.9, 0.7],
            [-0.6, 0.4, -0.8],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.7, 1.3, 0.8, 1.1, 0.9], dtype=jnp.float64)
    velocities = jnp.array(
        [
            [0.02, -0.01, 0.03],
            [-0.03, 0.04, -0.02],
            [0.01, 0.02, -0.01],
            [0.05, -0.02, 0.01],
            [-0.04, 0.03, 0.02],
            [0.03, 0.01, -0.04],
        ],
        dtype=jnp.float64,
    )
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=2,
    )
    vel_sorted = velocities[jnp.asarray(tree.particle_indices)]
    return tree, pos_sorted, mass_sorted, vel_sorted


def _build_multilevel_tree(n=600, leaf_size=8, seed=0):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(np.clip(rng.normal(0.0, 0.3, (n, 3)), -0.99, 0.99))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, n))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=pos.dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=pos.dtype),
    )
    tree, ps, msorted, _ = build_tree(
        pos, mass, bounds, return_reordered=True, leaf_size=leaf_size
    )
    return tree, ps, msorted


def test_static_num_levels_bit_identical_to_padded():
    """Passing the concrete (unpadded) depth must be bit-identical to deriving
    the M2M level count from the padded level_offsets shape."""
    from yggdrax.tree import get_num_levels

    tree, ps, ms = _build_multilevel_tree()
    actual_num_levels = int(get_num_levels(tree))

    padded = prepare_solidfmm_complex_upward_sweep(
        tree, ps, ms, max_order=4, center_mode="com"
    )
    optimized = prepare_solidfmm_complex_upward_sweep(
        tree,
        ps,
        ms,
        max_order=4,
        center_mode="com",
        static_num_levels=actual_num_levels,
    )
    # Exact equality: same arithmetic, only the empty padded levels are skipped.
    assert jnp.array_equal(padded.multipoles.packed, optimized.multipoles.packed)


def test_prepare_upward_sweep_stashes_and_reuses_concrete_depth():
    """The runtime method stashes the concrete depth and a traced (jitted) call
    reuses it, staying bit-identical to the concrete result."""
    import jax
    from yggdrax.tree import get_num_levels

    from jaccpot.runtime._fmm_impl import FastMultipoleMethod

    tree, ps, ms = _build_multilevel_tree()
    actual = int(get_num_levels(tree))
    fmm = FastMultipoleMethod(expansion_basis="solidfmm")

    concrete = fmm.prepare_upward_sweep(
        tree, ps, ms, max_order=4, center_mode="com", max_leaf_size=8
    )
    assert fmm._static_upward_num_levels == actual

    def _run(t, p, m):
        return fmm.prepare_upward_sweep(
            t, p, m, max_order=4, center_mode="com", max_leaf_size=8
        ).multipoles.packed

    packed_jit = jax.jit(_run)(tree, ps, ms)
    assert jnp.array_equal(packed_jit, concrete.multipoles.packed)


def test_prepare_solidfmm_upward_source_motion_optional_none():
    tree, pos_sorted, mass_sorted, _ = _build_sample_tree()
    upward = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
        center_mode="aabb",
    )
    assert upward.multipoles.source_motion_packed is None


def test_prepare_solidfmm_upward_source_motion_matches_finite_difference():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-6, dtype=pos_sorted.dtype)

    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
    )
    centers = base.multipoles.centers
    analytic = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        velocities_sorted=vel_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )

    assert analytic.multipoles.source_motion_packed is not None
    ref = (plus.multipoles.packed - minus.multipoles.packed) / (2.0 * dt)
    got = analytic.multipoles.source_motion_packed
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=2e-5, atol=1e-7)


def test_prepare_solidfmm_source_motion_multipoles_matches_upward_bundle():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
    )
    centers = base.multipoles.centers
    bundle = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        velocities_sorted=vel_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    direct = prepare_solidfmm_complex_source_motion_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        vel_sorted,
        max_order=order,
        centers=centers,
    )
    assert bundle.multipoles.source_motion_packed is not None
    assert np.allclose(
        np.asarray(direct),
        np.asarray(bundle.multipoles.source_motion_packed),
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_solidfmm_second_time_derivative_multipoles_matches_fd():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-5, dtype=pos_sorted.dtype)
    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
    )
    centers = base.multipoles.centers
    got = prepare_solidfmm_complex_source_motion_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        vel_sorted,
        max_order=order,
        centers=centers,
        time_derivative_order=2,
    )
    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    zero = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    ref = (
        plus.multipoles.packed - 2.0 * zero.multipoles.packed + minus.multipoles.packed
    ) / (dt * dt)
    rel = np.linalg.norm(np.asarray(got - ref)) / (
        np.linalg.norm(np.asarray(ref)) + 1e-12
    )
    assert rel < 2e-3


def _as_tree_upward_data(complex_upward) -> TreeUpwardData:
    multipoles = NodeMultipoleData(
        order=int(complex_upward.multipoles.order),
        centers=complex_upward.multipoles.centers,
        moments=None,  # type: ignore[arg-type]
        packed=complex_upward.multipoles.packed,
        component_matrix=complex_upward.multipoles.packed,
        source_motion_packed=complex_upward.multipoles.source_motion_packed,
    )
    return TreeUpwardData(
        geometry=complex_upward.geometry,
        mass_moments=complex_upward.mass_moments,
        multipoles=multipoles,
    )


def test_solidfmm_downward_source_motion_locals_match_finite_difference():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-6, dtype=pos_sorted.dtype)

    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
    )
    centers = base.multipoles.centers
    analytic = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        velocities_sorted=vel_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )

    fmm = FastMultipoleMethod(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(base),
        theta=0.6,
    )
    analytic_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(analytic),
        theta=0.6,
        interactions=base_down.interactions,
    )
    plus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(plus),
        theta=0.6,
        interactions=base_down.interactions,
    )
    minus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(minus),
        theta=0.6,
        interactions=base_down.interactions,
    )

    assert analytic_down.source_motion_locals is not None
    ref = (plus_down.locals.coefficients - minus_down.locals.coefficients) / (2.0 * dt)
    got = analytic_down.source_motion_locals.coefficients
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=3e-5, atol=1e-7)


def test_solidfmm_downward_second_time_derivative_locals_match_finite_difference():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-5, dtype=pos_sorted.dtype)

    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
    )
    centers = base.multipoles.centers
    d2m = prepare_solidfmm_complex_source_motion_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        vel_sorted,
        max_order=order,
        centers=centers,
        time_derivative_order=2,
    )

    fmm = FastMultipoleMethod(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(base),
        theta=0.6,
    )
    d2_upward = TreeUpwardData(
        geometry=base.geometry,
        mass_moments=base.mass_moments,
        multipoles=NodeMultipoleData(
            order=order,
            centers=centers,
            moments=None,  # type: ignore[arg-type]
            packed=d2m,
            component_matrix=d2m,
            source_motion_packed=None,
        ),
    )
    d2_down = fmm.prepare_downward_sweep(
        tree,
        d2_upward,
        theta=0.6,
        interactions=base_down.interactions,
    )

    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    zero = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    plus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(plus),
        theta=0.6,
        interactions=base_down.interactions,
    )
    zero_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(zero),
        theta=0.6,
        interactions=base_down.interactions,
    )
    minus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(minus),
        theta=0.6,
        interactions=base_down.interactions,
    )
    ref = (
        plus_down.locals.coefficients
        - 2.0 * zero_down.locals.coefficients
        + minus_down.locals.coefficients
    ) / (dt * dt)
    rel = np.linalg.norm(np.asarray(d2_down.locals.coefficients - ref)) / (
        np.linalg.norm(np.asarray(ref)) + 1e-12
    )
    assert rel < 2e-3
