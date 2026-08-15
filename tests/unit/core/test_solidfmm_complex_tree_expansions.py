"""Tests for solidfmm complex upward sweep helpers."""

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import build_tree

from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData


def _build_sample_tree(leaf_size=2):
    """The six-particle fixture.

    ``leaf_size`` is a parameter so the same particles can be built into a tree with
    internal nodes (the default, ``2``) or into a **single leaf** (any value >= 6),
    which is the ``num_internal_nodes == 0`` branch of
    ``_prepare_solidfmm_downward_sweep``. Keeping one fixture means the two regimes
    differ only in the tree, not in the physics.
    """
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
        leaf_size=leaf_size,
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

    from jaccpot.runtime._fmm_impl import FMMEngine

    tree, ps, ms = _build_multilevel_tree()
    actual = int(get_num_levels(tree))
    fmm = FMMEngine(expansion_basis="solidfmm")

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


#: theta for the two source-motion finite-difference tests below. **0.8, not 0.6, and
#: this is load-bearing.** At 0.6 this fixture accepts ZERO far pairs, so both tests
#: compared an all-zero `got` against an all-zero `ref` and could not have detected any
#: error in the source-motion M2L/L2L path -- D.12, in a different file. Measured at
#: 0.8: 4 far pairs, |ref| = 1.93, and the agreement is 1.08e-09 against a 3e-5 bound.
#: Both tests now assert the pair count and the reference magnitude so neither can
#: silently slide back into the vacuous regime.
_SOURCE_MOTION_THETA = 0.8

#: Smallest reference norm that counts as non-vacuous. |ref| is 1.93 at the settings
#: above, so this has three orders of headroom; it exists to fail loudly if a future
#: fixture or MAC change empties the interaction list again.
_NONVACUOUS_REF_NORM = 1e-3


def _assert_source_motion_comparison_is_nonvacuous(base_down, ref) -> None:
    """Fail if the far field is empty, before comparing anything against it.

    Both source-motion finite-difference tests below compare an analytic derivative
    against a central difference of the *locals*. With no accepted M2L pairs both
    sides are identically zero and the comparison holds for free -- which is the state
    they were both in. Asserting the pair count and the reference magnitude is what
    makes the agreement mean something.
    """
    far_pairs = int(np.asarray(base_down.interactions.sources).shape[0])
    assert far_pairs > 0, (
        f"no M2L pairs accepted at theta={_SOURCE_MOTION_THETA}, so the source-motion "
        f"far field is not exercised and this comparison is vacuous"
    )
    ref_norm = float(np.linalg.norm(np.asarray(ref)))
    assert ref_norm > _NONVACUOUS_REF_NORM, (
        f"finite-difference reference norm {ref_norm:.3e} is below "
        f"{_NONVACUOUS_REF_NORM:.0e}; an all-but-zero reference would be matched by "
        f"an all-but-zero result regardless of correctness"
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

    fmm = FMMEngine(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(base),
        theta=_SOURCE_MOTION_THETA,
    )
    analytic_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(analytic),
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )
    plus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(plus),
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )
    minus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(minus),
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )

    assert analytic_down.source_motion_locals is not None
    ref = (plus_down.locals.coefficients - minus_down.locals.coefficients) / (2.0 * dt)
    got = analytic_down.source_motion_locals.coefficients
    _assert_source_motion_comparison_is_nonvacuous(base_down, ref)
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=3e-5, atol=1e-7)


def test_solidfmm_downward_source_motion_locals_when_the_tree_has_no_internal_nodes():
    """The single-leaf tree: source-motion locals exist and are exactly zero.

    Zero is the *right* answer, not merely a safe one. A tree with no internal nodes
    is a single leaf, so there are no box pairs, the M2L interaction list is empty and
    there is no parent-to-child cascade. The whole interaction is near-field, which
    this function does not compute.

    **Which line this covers.** It exercises the ``pair_count == 0`` early return in
    ``_prepare_solidfmm_downward_sweep``, *not* the ``else`` of
    ``num_internal_nodes > 0``. That matters, because the ``else`` is where audit
    G.1a's latent ``NameError`` lived, and this test does **not** reach it: getting
    there needs ``pair_count > 0`` *and* ``num_internal_nodes == 0``, and the
    ``pair_count == 0`` return has already claimed every single-leaf tree. Measured
    while writing this -- even forcing a one-node tree with a foreign interaction list
    does not reach it. G.1a is dead code, and the audit's reachability note missed
    that guard. What this test pins is the reachable half of the same contract.

    ``is not None`` is asserted separately and deliberately: ``None`` means "no source
    motion was requested", so returning it for "requested, and the far field is empty"
    would conflate two states the consumer cannot tell apart.
    """
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree(leaf_size=8)
    # The branch under test. Without this the test would silently drift into the
    # `num_internal_nodes > 0` side and pass for the wrong reason.
    assert int(jnp.asarray(tree.left_child).shape[0]) == 0

    order = 4
    dt = jnp.asarray(1e-6, dtype=pos_sorted.dtype)

    base = prepare_solidfmm_complex_upward_sweep(
        tree, pos_sorted, mass_sorted, max_order=order, center_mode="aabb"
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

    fmm = FMMEngine(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(tree, _as_tree_upward_data(base), theta=0.6)
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
    # The complement of the guard the other two tests use: here an empty far field is
    # the regime under test, so pin it rather than reject it.
    assert int(np.asarray(base_down.interactions.sources).shape[0]) == 0

    got = np.asarray(analytic_down.source_motion_locals.coefficients)
    ref = np.asarray(
        (plus_down.locals.coefficients - minus_down.locals.coefficients) / (2.0 * dt)
    )
    assert got.shape == np.asarray(base_down.locals.coefficients).shape
    np.testing.assert_array_equal(got, np.zeros_like(got))
    np.testing.assert_array_equal(ref, np.zeros_like(ref))


def test_solidfmm_downward_second_time_derivative_locals_match_finite_difference():
    """The analytic d2/dt2 locals against a central second difference.

    ``dt = 1e-3``, **not** ``1e-5``, and the bound is ``1e-5`` rather than ``2e-3``.
    Both changed for the same reason: with a non-empty far field (see
    ``_SOURCE_MOTION_THETA``) the old settings do not pass, and the reason is the
    *reference*, not the code. A central second difference carries roundoff of order
    ``eps / dt**2``, so small ``dt`` is actively harmful here -- the optimum is near
    ``eps ** 0.25 ~ 1.2e-4``. Measured, holding everything else fixed:

        dt = 1e-6   rel = 4.0e-01
        dt = 1e-5   rel = 3.0e-03      <- the old setting, over the old 2e-3 bound
        dt = 1e-4   rel = 3.9e-05
        dt = 1e-3   rel = 2.5e-07      <- chosen
        dt = 1e-2   rel = 8.7e-09

    Monotone improvement as ``dt`` grows is the signature of a roundoff-dominated
    difference; the multipoles are polynomial in the displacement, so truncation stays
    negligible over this whole range. ``1e-3`` sits in the flat region with ~40x
    headroom under the new ``1e-5`` bound.

    So this is a strengthening on both axes -- the comparison is no longer vacuous and
    the bound is 200x tighter -- and the old ``2e-3`` was never a real bound: it was
    calibrated in the regime where ``rel`` was identically zero.
    """
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-3, dtype=pos_sorted.dtype)

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

    fmm = FMMEngine(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(base),
        theta=_SOURCE_MOTION_THETA,
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
        theta=_SOURCE_MOTION_THETA,
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
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )
    zero_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(zero),
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )
    minus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(minus),
        theta=_SOURCE_MOTION_THETA,
        interactions=base_down.interactions,
    )
    ref = (
        plus_down.locals.coefficients
        - 2.0 * zero_down.locals.coefficients
        + minus_down.locals.coefficients
    ) / (dt * dt)
    # Before the ratio: the `+ 1e-12` below makes an all-zero reference give rel == 0,
    # i.e. an automatic pass. Assert the far field is actually there.
    _assert_source_motion_comparison_is_nonvacuous(base_down, ref)
    rel = np.linalg.norm(np.asarray(d2_down.locals.coefficients - ref)) / (
        np.linalg.norm(np.asarray(ref)) + 1e-12
    )
    assert rel < 1e-5, (
        f"analytic d2/dt2 locals vs central second difference: rel {rel:.3e}. "
        f"See this test's docstring for the dt sweep that sets the 1e-5 bound."
    )
