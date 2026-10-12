"""Jerk and the time-derivative tower, and the source-motion multipoles under them.

``prepare_real_source_motion_multipoles`` returns ``d^k M / dt^k`` for every node
when the particles move on straight lines about frozen centres. These tests use a
basis-independent oracle -- central finite differences of the ordinary real
multipoles at those same centres. The engine-level check, with the far field
engaged, is against direct sums:
``test_real_source_motion_with_the_far_field_matches_direct_sum``.

The public jerk and time-derivative APIs (``compute_accelerations_and_jerk``,
``compute_accelerations_with_time_derivatives``, ``max_acc_derivative_order``) and
the direct-sum jerk / snap / crackle oracles came here from ``test_solver_api.py``
in the 2026-10 cleanup (Z-a); this file was ``test_real_source_motion.py``. Only
the source-motion tests need float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.tree import get_level_offsets, get_nodes_by_level

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMPreset
from jaccpot.runtime._level_shapes import level_batch_width
from jaccpot.upward.real_tree_expansions import (
    _p2m_leaves_real,
    aggregate_m2m_real_by_level,
    prepare_real_source_motion_multipoles,
)

_NEEDS_X64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="finite differences need float64"
)

ORDER = 4


def _problem(n=96, seed=11):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(scale=0.6, size=(n, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    velocities = jnp.asarray(rng.normal(scale=0.3, size=(n, 3)))
    return positions, masses, velocities


def _frozen(state):
    tree = state.tree
    return dict(
        tree=tree,
        centers=jnp.asarray(state.upward.multipoles.centers),
        total=int(jnp.asarray(tree.parent).shape[0]),
        internal=int(jnp.asarray(tree.left_child).shape[0]),
        leaf=int(state.max_leaf_size),
    )


def _multipoles_at(f, positions_sorted, masses_sorted):
    """The ordinary real P2M + M2M at the FROZEN centres of ``f``."""
    tree = f["tree"]
    offsets = get_level_offsets(tree)
    packed = _p2m_leaves_real(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        f["centers"],
        order=ORDER,
        max_leaf_size=f["leaf"],
        num_internal=f["internal"],
        total_nodes=f["total"],
        leaf_batch_size=64,
    )
    return aggregate_m2m_real_by_level(
        packed,
        f["centers"],
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(get_nodes_by_level(tree), dtype=INDEX_DTYPE),
        jnp.asarray(offsets, dtype=INDEX_DTYPE),
        order=ORDER,
        num_internal=f["internal"],
        num_levels=max(int(offsets.shape[0] - 1), 1),
        level_batch_width=level_batch_width(
            offsets, total_nodes=f["total"], num_internal=f["internal"]
        ),
    )


@_NEEDS_X64
def test_source_motion_multipoles_are_the_time_derivatives():
    positions, masses, velocities = _problem()
    fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    state = fmm.prepare_state(positions, masses, max_order=ORDER, leaf_size=4)
    f = _frozen(state)
    order_idx = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
    x = jnp.asarray(state.positions_sorted)
    m = jnp.asarray(state.masses_sorted)
    v = velocities[order_idx]

    def moved(t):
        return _multipoles_at(f, x + t * v, m)

    h = 1e-3
    m0, mp, mm = moved(0.0), moved(h), moved(-h)
    fd1 = (mp - mm) / (2.0 * h)
    fd2 = (mp - 2.0 * m0 + mm) / (h * h)
    for k, fd in ((1, fd1), (2, fd2)):
        got = prepare_real_source_motion_multipoles(
            f["tree"],
            x,
            m,
            v,
            max_order=ORDER,
            centers=f["centers"],
            time_derivative_order=k,
            max_leaf_size=f["leaf"],
        )
        err = float(jnp.linalg.norm(got - fd) / jnp.linalg.norm(got))
        # central differences: O(h^2) truncation, plus round-off / h^k
        assert err < (1e-6 if k == 1 else 1e-4), f"k={k}: rel {err:.3e}"


@_NEEDS_X64
def test_a_resting_system_has_no_source_motion():
    positions, masses, _ = _problem(n=40)
    fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    state = fmm.prepare_state(positions, masses, max_order=ORDER, leaf_size=4)
    f = _frozen(state)
    got = prepare_real_source_motion_multipoles(
        f["tree"],
        jnp.asarray(state.positions_sorted),
        jnp.asarray(state.masses_sorted),
        jnp.zeros_like(state.positions_sorted),
        max_order=ORDER,
        centers=f["centers"],
        time_derivative_order=1,
        max_leaf_size=f["leaf"],
    )
    assert float(jnp.max(jnp.abs(got))) == 0.0


@_NEEDS_X64
def test_real_source_motion_with_the_far_field_matches_direct_sum():
    """Jerk and the K = 3 tower with the far field engaged, against direct sums.

    The engine-level owner of the far-field time derivatives: the direct-sum tests
    in ``test_solver_api.py`` run the tower at ``theta=1e-4`` (no M2L pair), and
    ``test_jerk_modes_diverge_against_direct_sum_when_far_field_engaged`` covers
    the jerk only. Here snap and crackle go through the far field too.

    Replaces ``test_real_source_motion_matches_complex`` (2026-10 cleanup, C), which
    compared the same quantities with the complex basis to 1e-12. Measured at
    N = 96, leaf 4, p = 4, theta 0.6 (106 M2L pairs), fp64, Plummer softening 1e-3
    (the direct sums' kernel), relative L2 against the direct sums -- identical on
    the complex basis before it was removed:

        jerk     4.338e-4   (both entry points)
        snap     8.285e-5
        crackle  9.554e-6

    The bounds give each ~2.3-3x headroom. The two jerk entry points
    (``compute_accelerations_and_jerk`` and the tower's first level) run the same
    analytic path and must agree to round-off.
    """
    positions, masses = _sample_problem(n=96)
    velocities = _sample_velocities(n=96)
    positions, masses, velocities = (
        jnp.asarray(a, jnp.float64) for a in (positions, masses, velocities)
    )
    kwargs = dict(leaf_size=4, max_order=ORDER, theta=0.6)

    def solver():
        return FastMultipoleMethod(
            preset=FMMPreset.ACCURATE, softening=1e-3, softening_kernel="plummer"
        )

    state = solver().prepare_state(positions, masses, **kwargs)
    far_pairs = int(jnp.sum(state.interactions.counts))
    assert far_pairs > 0, f"the far field must be engaged (got {far_pairs} M2L pairs)"

    _, jerk = solver().compute_accelerations_and_jerk(
        positions, masses, velocities, jerk_mode="accurate", **kwargs
    )
    _, derivs = solver().compute_accelerations_with_time_derivatives(
        positions, masses, velocities, max_time_derivative_order=3, **kwargs
    )
    assert len(derivs) == 3
    jerk_gap = float(jnp.linalg.norm(jerk - derivs[0]) / jnp.linalg.norm(derivs[0]))
    assert jerk_gap < 1e-12, jerk_gap

    references = [
        oracle(positions, masses, velocities, G=1.0, softening=1e-3)
        for oracle in (_direct_sum_jerk, _direct_sum_snap, _direct_sum_crackle)
    ]
    bounds = (1e-3, 2.5e-4, 3e-5)
    for name, got, want, bound in zip(
        ("jerk", "snap", "crackle"), derivs, references, bounds
    ):
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < bound, f"{name}: rel-L2 {rel:.3e} >= {bound:.1e}"


# ---------------------------------------------------------------------------
# From test_solver_api.py (2026-10 cleanup, Z-a): jerk and the time-derivative APIs
# ---------------------------------------------------------------------------


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(11)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def _sample_velocities(n: int = 64):
    key = jax.random.PRNGKey(17)
    return jax.random.uniform(
        key,
        (n, 3),
        minval=-0.2,
        maxval=0.2,
        dtype=jnp.float32,
    )


def _direct_sum_jerk(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    velocities: jnp.ndarray,
    *,
    G: float,
    softening: float,
) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    vdiff = velocities[:, None, :] - velocities[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening**2
    eps = jnp.finfo(positions.dtype).eps
    inv_r = jnp.where(dist_sq > 0, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_r3 = inv_r / dist_sq
    inv_r5 = inv_r3 / dist_sq
    eye = jnp.eye(positions.shape[0], dtype=bool)
    inv_r3 = jnp.where(eye, 0.0, inv_r3)
    inv_r5 = jnp.where(eye, 0.0, inv_r5)
    rv = jnp.sum(diff * vdiff, axis=-1)
    term = vdiff * inv_r3[..., None] - 3.0 * rv[..., None] * diff * inv_r5[..., None]
    weighted = masses[None, :, None] * term
    return -G * jnp.sum(weighted, axis=1)


def _direct_sum_snap(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    velocities: jnp.ndarray,
    *,
    G: float,
    softening: float,
) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    vdiff = velocities[:, None, :] - velocities[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening**2
    eps = jnp.finfo(positions.dtype).eps
    inv_r = jnp.where(dist_sq > 0, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_r3 = inv_r / dist_sq
    inv_r5 = inv_r3 / dist_sq
    inv_r7 = inv_r5 / dist_sq
    eye = jnp.eye(positions.shape[0], dtype=bool)
    inv_r3 = jnp.where(eye, 0.0, inv_r3)
    inv_r5 = jnp.where(eye, 0.0, inv_r5)
    inv_r7 = jnp.where(eye, 0.0, inv_r7)
    rv = jnp.sum(diff * vdiff, axis=-1)
    vv = jnp.sum(vdiff * vdiff, axis=-1)
    term = (
        6.0 * rv[..., None] * vdiff * inv_r5[..., None]
        + 3.0 * vv[..., None] * diff * inv_r5[..., None]
        - 15.0 * (rv * rv)[..., None] * diff * inv_r7[..., None]
    )
    weighted = masses[None, :, None] * term
    return G * jnp.sum(weighted, axis=1)


def _direct_sum_crackle(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    velocities: jnp.ndarray,
    *,
    G: float,
    softening: float,
) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    vdiff = velocities[:, None, :] - velocities[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening**2
    eps = jnp.finfo(positions.dtype).eps
    inv_r = jnp.where(dist_sq > 0, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_r3 = inv_r / dist_sq
    inv_r5 = inv_r3 / dist_sq
    inv_r7 = inv_r5 / dist_sq
    inv_r9 = inv_r7 / dist_sq
    eye = jnp.eye(positions.shape[0], dtype=bool)
    inv_r5 = jnp.where(eye, 0.0, inv_r5)
    inv_r7 = jnp.where(eye, 0.0, inv_r7)
    inv_r9 = jnp.where(eye, 0.0, inv_r9)
    rv = jnp.sum(diff * vdiff, axis=-1)
    vv = jnp.sum(vdiff * vdiff, axis=-1)
    term = (
        9.0 * vv[..., None] * vdiff * inv_r5[..., None]
        - 45.0 * (rv * rv)[..., None] * vdiff * inv_r7[..., None]
        - 45.0 * rv[..., None] * vv[..., None] * diff * inv_r7[..., None]
        + 105.0 * (rv * rv * rv)[..., None] * diff * inv_r9[..., None]
    )
    weighted = masses[None, :, None] * term
    return G * jnp.sum(weighted, axis=1)


def test_compute_accelerations_returns_acc_derivatives_when_requested():
    positions, masses = _sample_problem(n=48)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
    )
    acc, derivatives = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    assert acc.shape == positions.shape
    assert len(derivatives) == 1
    assert derivatives[0].shape == (positions.shape[0], 3, 3)


# The near-field-only (theta=1e-4) jerk and time-derivative tests below share one
# configuration, N=20 at leaf 10 (2026-10-12: they had four, (20, 12), (16, 10),
# (20, 10) and (12, 8)), so one worker compiles its tree once. Measured there:
# jerk rel-L2 4.3e-5 in both modes, snap 6.1e-5, crackle 7.3e-5 -- against
# bounds of 5e-2, 2e-3, 5e-3, 3e-2 and 6e-2.
_NEAR_ONLY_N = 20
_NEAR_ONLY_LEAF = 10


def test_compute_accelerations_and_jerk_matches_direct_sum_small_n():
    """Jerk matches direct summation in the NEAR-FIELD-ONLY regime.

    ``theta=1e-4`` accepts **no** M2L pairs at this N, so this is a
    direct-summation self-check: it bounds the near-field jerk and the
    L2P/evaluation plumbing, and says nothing about the far field. The error
    (4.3e-5 measured) is identical for both ``jerk_mode`` values -- which is the
    tell that the far-field source-motion term never runs here. Far-field coverage
    is :func:`test_jerk_modes_diverge_against_direct_sum_when_far_field_engaged`.
    """
    n = _NEAR_ONLY_N
    positions, masses = _sample_problem(n=n)
    velocities = _sample_velocities(n=n)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.ACCURATE,
        basis="real",
    )
    acc_fmm, jerk_fmm = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        leaf_size=_NEAR_ONLY_LEAF,
        max_order=4,
        theta=1e-4,
    )
    jerk_ref = _direct_sum_jerk(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    rel = np.linalg.norm(np.asarray(jerk_fmm - jerk_ref)) / (
        np.linalg.norm(np.asarray(jerk_ref)) + 1e-12
    )
    assert acc_fmm.shape == positions.shape
    assert jerk_fmm.shape == positions.shape
    assert rel < 5e-2


def test_compute_accelerations_and_jerk_accurate_mode_matches_direct_sum_tighter(
    monkeypatch: pytest.MonkeyPatch,
):
    """``jerk_mode="accurate"`` matches direct summation, near-field-only regime.

    Same caveat as
    :func:`test_compute_accelerations_and_jerk_matches_direct_sum_small_n`:
    ``theta=1e-4`` accepts no M2L pairs, so the tighter 2e-3 bound here does not
    reflect the ``"accurate"`` scheme doing more work. The two tolerances in these
    two tests are NOT a measurement of what ``"fast_approx"`` gives up; see
    :func:`test_jerk_modes_diverge_against_direct_sum_when_far_field_engaged`.

    The same call also pins how the accurate mode gets there, which were two
    tests of their own on the identical call until 2026-10-12: it reuses the
    prepared topology (no full rebuild solve through ``compute_accelerations``)
    and, on the real (solidfmm) basis, takes the analytic path (no
    finite-difference fallback through
    ``_evaluate_prepared_state_at_positions_sorted``). Both are forbidden below.
    """
    n = _NEAR_ONLY_N
    positions, masses = _sample_problem(n=n)
    velocities = _sample_velocities(n=n)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.ACCURATE,
        basis="real",
    )

    def _forbidden_rebuild(*args, **kwargs):
        raise AssertionError(
            "accurate jerk mode should reuse prepared topology and avoid full rebuild solves"
        )

    def _forbidden_fd_fallback(*args, **kwargs):
        raise AssertionError(
            "solidfmm accurate jerk should not use finite-difference fallback evaluations"
        )

    monkeypatch.setattr(fmm._impl, "compute_accelerations", _forbidden_rebuild)
    monkeypatch.setattr(
        fmm._impl,
        "_evaluate_prepared_state_at_positions_sorted",
        _forbidden_fd_fallback,
    )
    acc, jerk_acc = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        leaf_size=_NEAR_ONLY_LEAF,
        max_order=4,
        theta=1e-4,
        jerk_mode="accurate",
    )
    assert acc.shape == positions.shape
    assert jerk_acc.shape == positions.shape
    jerk_ref = _direct_sum_jerk(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    err_acc = np.linalg.norm(np.asarray(jerk_acc - jerk_ref)) / (
        np.linalg.norm(np.asarray(jerk_ref)) + 1e-12
    )
    assert err_acc < 2e-3


def _far_pair_count(state) -> int:
    """M2L interactions in a prepared topology; 0 means the far field never runs."""
    interactions = state.interactions
    if interactions is not None:
        return int(jnp.sum(interactions.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


def test_jerk_modes_diverge_against_direct_sum_when_far_field_engaged():
    """``"accurate"`` beats ``"fast_approx"`` against direct summation -- measured.

    The two direct-sum jerk tests above both run at ``theta=1e-4``, which accepts
    no M2L pairs, so neither exercises the far-field source-motion term that is the
    only difference between the modes. This does, and it is the test that actually
    bounds what ``"fast_approx"`` gives up.

    Configuration matters and is asserted, not assumed: ``leaf_size=4`` is what
    makes the MAC accept anything at this N. At ``leaf_size=10`` (what the tests
    above use) or 12 the far-pair count is **zero** and both modes return
    bit-identical jerks, so a version of this test with the usual leaf size would
    pass while measuring nothing. Hence the ``far_pairs > 0`` guard.

    Measured at N=96, leaf 4, p=4, ``theta=0.6`` (106 M2L pairs), relative L2
    against direct summation:

        fast_approx  1.74e-03
        accurate     4.34e-04     -> 4.0x better

    The ratio is the load-bearing assertion. Across N in {48, 96, 256} and theta in
    {0.4, 0.5, 0.6} it ranges 2.8x-8.2x, and it collapses to exactly 1.0x whenever
    the far-pair count is 0 -- so it is a direct probe of the source-motion term
    being present. A floor of 2.0 sits below the measured minimum with margin.
    Absolute bounds are set from the measurement plus roughly 3x headroom for the
    fp32 sample; they are not tight convergence bounds and should not be read as
    such.
    """
    n = 96
    leaf_size = 4
    order = 4
    theta = 0.6

    positions, masses = _sample_problem(n=n)
    velocities = _sample_velocities(n=n)

    # Anti-vacuity guard: the same parameters must actually accept M2L pairs.
    probe = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    far_pairs = _far_pair_count(
        probe.prepare_state(
            positions, masses, max_order=order, leaf_size=leaf_size, theta=theta
        )
    )
    assert far_pairs > 0, (
        "config must engage the far field or this test measures nothing "
        f"(got {far_pairs} M2L pairs; leaf_size=10 or 12 gives 0 at this N)"
    )

    jerk_ref = _direct_sum_jerk(positions, masses, velocities, G=1.0, softening=1e-3)
    ref_norm = np.linalg.norm(np.asarray(jerk_ref)) + 1e-12

    errors = {}
    for mode in ("fast_approx", "accurate"):
        fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
        kwargs = dict(leaf_size=leaf_size, max_order=order, theta=theta, jerk_mode=mode)
        _, jerk = fmm.compute_accelerations_and_jerk(
            positions, masses, velocities, **kwargs
        )
        errors[mode] = float(np.linalg.norm(np.asarray(jerk - jerk_ref)) / ref_norm)

    assert errors["fast_approx"] < 6e-3, errors
    assert errors["accurate"] < 2e-3, errors

    improvement = errors["fast_approx"] / max(errors["accurate"], 1e-30)
    assert improvement > 2.0, (
        "jerk_mode='accurate' should be materially closer to direct summation "
        f"than 'fast_approx' once the far field is engaged, but improved only "
        f"{improvement:.2f}x (fast={errors['fast_approx']:.3e}, "
        f"accurate={errors['accurate']:.3e}). A ratio near 1.0 means the analytic "
        "far-field source-motion term is not contributing."
    )


def test_compute_accelerations_and_jerk_invalid_mode_raises():
    positions, masses = _sample_problem(n=16)
    velocities = _sample_velocities(n=16)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    with pytest.raises(ValueError, match="jerk_mode"):
        _ = fmm.compute_accelerations_and_jerk(
            positions,
            masses,
            velocities,
            jerk_mode="nope",
        )


def test_compute_accelerations_with_time_derivatives_k3_matches_direct_sum():
    n = _NEAR_ONLY_N
    positions, masses = _sample_problem(n=n)
    velocities = _sample_velocities(n=n)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.ACCURATE,
        basis="real",
    )
    acc, derivs = fmm.compute_accelerations_with_time_derivatives(
        positions,
        masses,
        velocities,
        leaf_size=_NEAR_ONLY_LEAF,
        max_order=4,
        theta=1e-4,
        max_time_derivative_order=3,
        mode="accurate",
    )
    assert acc.shape == positions.shape
    assert len(derivs) == 3
    jerk_ref = _direct_sum_jerk(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    snap_ref = _direct_sum_snap(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    crackle_ref = _direct_sum_crackle(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    err_jerk = np.linalg.norm(np.asarray(derivs[0] - jerk_ref)) / (
        np.linalg.norm(np.asarray(jerk_ref)) + 1e-12
    )
    err_snap = np.linalg.norm(np.asarray(derivs[1] - snap_ref)) / (
        np.linalg.norm(np.asarray(snap_ref)) + 1e-12
    )
    err_crackle = np.linalg.norm(np.asarray(derivs[2] - crackle_ref)) / (
        np.linalg.norm(np.asarray(crackle_ref)) + 1e-12
    )
    assert err_jerk < 5e-3
    assert err_snap < 3e-2
    assert err_crackle < 6e-2
