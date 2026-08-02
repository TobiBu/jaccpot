"""Correctness gate for the momentum-conserving (mutual) FMM.

Two properties are asserted here and they have very different tolerances, which
is the whole point of the mutual restructure:

* the **force** stays FMM-approximate -- accuracy is set by ``theta`` and the
  expansion order, checked against the exact direct sum at ~1e-3 .. 1e-5;
* the **momentum** becomes exact -- ``sum_i m_i a_i`` cancels to round-off
  (~1e-16 relative), *independently* of ``theta`` and order.

The second is what a block-step individual-timestep integrator needs and what
jaccpot's production target-centric force cannot supply: there each pair is
evaluated twice, independently, so the momentum residual sits at the force
accuracy rather than at round-off.

Several tests deliberately configure a **multi-level** tree (large ``theta``,
small ``leaf_size``) so the MAC accepts a non-empty far list, and assert
``num_far_pairs > 0``. Without that the FMM degenerates to the near-field direct
sum, matches to 1e-16 for trivial reasons, and every far-field assertion below
becomes vacuous.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual import (
    active_level_floor,
    boundary_level_weights,
    build_mutual_state,
    build_mutual_topology,
    is_sync_boundary,
    mutual_accelerations,
    mutual_level_accelerations,
    mutual_weighted_accelerations,
    n_sub,
)
from jaccpot.nornax_adapter import BlockStepFMM

SOFTENING = 1.0e-2


def _system(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(0.0, scale, (n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=jnp.float64)
    return positions, masses


def _direct_sum_accelerations(positions, masses, *, softening=SOFTENING, G=1.0):
    """Exact O(N^2) reference, built the momentum-conserving way.

    ``c_ij`` is symmetric to the last bit and ``dr_ij`` is the exact negation of
    ``dr_ji``, so this reference itself conserves momentum structurally -- the
    same construction nornax's ``MutualDirectSumGravity`` uses.
    """
    dr = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + softening**2
    live = ~jnp.eye(positions.shape[0], dtype=bool)
    safe = jnp.where(live, r2, 1.0)
    inv_r3 = jnp.where(live, safe ** (-1.5), 0.0)
    c = G * masses[:, None] * masses[None, :] * inv_r3
    return jnp.sum(c[..., None] * dr, axis=1) / masses[:, None]


def _level_oracle(positions, masses, rung, level, *, softening=SOFTENING, G=1.0):
    """Dense per-particle level-``k`` reference (nornax's oracle, inlined).

    Inlined rather than imported so the level algebra is checked even where
    nornax cannot be imported; the cross-repo module asserts this matches
    nornax's own implementation.
    """
    dr = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + softening**2
    live = ~jnp.eye(positions.shape[0], dtype=bool)
    live = live & (jnp.maximum(rung[:, None], rung[None, :]) == level)
    safe = jnp.where(live, r2, 1.0)
    inv_r3 = jnp.where(live, safe ** (-1.5), 0.0)
    c = G * masses[:, None] * masses[None, :] * inv_r3
    return jnp.sum(c[..., None] * dr, axis=1) / masses[:, None]


def _momentum_residual(accelerations, masses):
    """``|sum_i m_i a_i|`` normalised by the scale of the terms being summed."""
    terms = masses[:, None] * accelerations
    scale = jnp.sum(jnp.abs(terms), axis=0)
    return float(
        jnp.linalg.norm(jnp.sum(terms, axis=0)) / (jnp.linalg.norm(scale) + 1e-300)
    )


def _build(
    n,
    *,
    theta,
    order,
    leaf_size,
    seed=0,
    softening=SOFTENING,
    backend="jax",
    interpret=False,
):
    positions, masses = _system(n, seed=seed)
    topology, _ = build_mutual_topology(
        positions, masses, theta=theta, order=order, leaf_size=leaf_size
    )
    state = build_mutual_state(
        topology,
        softening=softening,
        use_pallas=(backend == "pallas"),
        pallas_interpret=bool(interpret),
    )
    return positions, masses, topology, state


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, leaf_size, theta", [(256, 8, 0.6), (512, 16, 0.9), (600, 16, 1.2)]
)
def test_dual_traversal_covers_every_pair_exactly_once(n, leaf_size, theta):
    """The near/far partition must be a partition -- no gaps, no double counts.

    Mutual accumulation makes double counting invisible in the momentum check (a
    doubled pair still cancels), so it has to be caught structurally here.
    """
    positions, masses = _system(n)
    topology, _ = build_mutual_topology(
        positions, masses, theta=theta, order=4, leaf_size=leaf_size
    )
    ranges = topology.node_particle_ranges
    count = np.zeros((n, n), dtype=np.int32)

    def members(node):
        start, end = ranges[node]
        return np.arange(start, end + 1)

    for a, b in zip(topology.far_a, topology.far_b):
        ia, ib = members(a), members(b)
        count[np.ix_(ia, ib)] += 1
        count[np.ix_(ib, ia)] += 1
    for a, b in zip(topology.near_a, topology.near_b):
        ia, ib = members(a), members(b)
        count[np.ix_(ia, ib)] += 1
        count[np.ix_(ib, ia)] += 1
    for leaf in topology.leaf_nodes:
        il = members(leaf)
        count[np.ix_(il, il)] += 1

    off_diagonal = count[~np.eye(n, dtype=bool)]
    assert off_diagonal.min() == 1
    assert off_diagonal.max() == 1


def test_canonical_lists_are_strictly_ordered():
    """Every emitted pair is canonical ``a < b``; that is what prevents doubles."""
    positions, masses = _system(512)
    topology, _ = build_mutual_topology(
        positions, masses, theta=0.9, order=4, leaf_size=16
    )
    assert topology.num_far_pairs > 0
    assert np.all(topology.far_a < topology.far_b)
    assert np.all(topology.near_a < topology.near_b)


# ---------------------------------------------------------------------------
# Phase 1 -- mutual near field
# ---------------------------------------------------------------------------


def test_near_field_only_matches_direct_sum_to_round_off():
    """With an empty far list the FMM *is* the direct sum -- to 1e-14, not 1e-3.

    This isolates the symmetric P2P kernel: any bug in the ``+F``/``-F`` scatter
    shows up immediately, with no truncation error to hide behind.
    """
    positions, masses, topology, state = _build(512, theta=0.3, order=4, leaf_size=32)
    assert topology.num_far_pairs == 0
    accelerations = mutual_accelerations(state, positions, masses)
    reference = _direct_sum_accelerations(positions, masses)
    error = float(
        jnp.linalg.norm(accelerations - reference) / jnp.linalg.norm(reference)
    )
    assert error < 1e-13
    assert _momentum_residual(accelerations, masses) < 1e-14


# ---------------------------------------------------------------------------
# Phase 2 -- mutual far field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta, order, tolerance",
    [(0.7, 4, 3e-3), (0.7, 6, 3e-3), (1.0, 4, 3e-2), (0.9, 6, 1e-2)],
)
def test_total_force_matches_direct_sum_within_fmm_tolerance(theta, order, tolerance):
    positions, masses, topology, state = _build(
        2048, theta=theta, order=order, leaf_size=16
    )
    assert topology.num_far_pairs > 0
    accelerations = mutual_accelerations(state, positions, masses)
    reference = _direct_sum_accelerations(positions, masses)
    error = float(
        jnp.linalg.norm(accelerations - reference) / jnp.linalg.norm(reference)
    )
    assert error < tolerance
    # Bound below too: a vacuous far list would make this pass trivially.
    assert error > 1e-12


@pytest.mark.parametrize("theta", [0.5, 0.7, 1.0])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_momentum_is_exact_independently_of_theta_and_order(theta, order, backend):
    """The defining property: momentum error is round-off, not truncation error.

    Sweeping ``theta`` and the order changes the *force* error by orders of
    magnitude while the momentum residual must stay pinned at round-off. A
    non-mutual FMM fails this immediately -- its residual tracks the force error.

    The Pallas backend runs here in interpret mode, so the real kernel logic is
    executed on CPU CI. That matters more for this test than for any other: the
    mutual P2P kernel's whole reason to exist is that it computes ``dr`` once and
    *negates* it, and a port that recomputed ``dr`` for the source-leaf pass
    would produce forces that pass every accuracy assertion in this file while
    moving this residual from 1e-17 to the force accuracy. Hence the 1e-13 bound
    rather than anything resembling a force tolerance.
    """
    positions, masses, topology, state = _build(
        1024,
        theta=theta,
        order=order,
        leaf_size=16,
        backend=backend,
        interpret=(backend == "pallas"),
    )
    accelerations = mutual_accelerations(state, positions, masses)
    assert _momentum_residual(accelerations, masses) < 1e-13


def test_momentum_residual_beats_force_error_by_orders_of_magnitude():
    """Guards against a regression that silently reverts to a gather kernel."""
    positions, masses, topology, state = _build(2048, theta=1.0, order=2, leaf_size=16)
    assert topology.num_far_pairs > 0
    accelerations = mutual_accelerations(state, positions, masses)
    reference = _direct_sum_accelerations(positions, masses)
    force_error = float(
        jnp.linalg.norm(accelerations - reference) / jnp.linalg.norm(reference)
    )
    momentum = _momentum_residual(accelerations, masses)
    assert force_error > 1e-4
    assert momentum < force_error * 1e-8


# ---------------------------------------------------------------------------
# Phase 3 -- rung awareness
# ---------------------------------------------------------------------------


K_MAX = 3


def _rungs(n, k_max=K_MAX, seed=11):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, k_max + 1, n), dtype=jnp.int32)


def test_levels_partition_the_total_force():
    """``sum_k a_k`` must reproduce the full acceleration exactly.

    This is what makes the level split a *partition*: nornax sums the levels to
    recover the total, so any leakage or double count corrupts every step.
    """
    positions, masses, _, state = _build(1024, theta=0.9, order=4, leaf_size=16)
    rung = _rungs(1024)
    total = mutual_accelerations(state, positions, masses)
    summed = sum(
        mutual_level_accelerations(
            state, positions, masses, rung=rung, level=k, k_max=K_MAX
        )
        for k in range(K_MAX + 1)
    )
    error = float(jnp.linalg.norm(summed - total) / jnp.linalg.norm(total))
    assert error < 1e-13


@pytest.mark.parametrize("level", list(range(K_MAX + 1)))
def test_each_level_conserves_momentum(level):
    positions, masses, _, state = _build(1024, theta=0.9, order=4, leaf_size=16)
    rung = _rungs(1024)
    accelerations = mutual_level_accelerations(
        state, positions, masses, rung=rung, level=level, k_max=K_MAX
    )
    assert jnp.linalg.norm(accelerations) > 0.0
    assert _momentum_residual(accelerations, masses) < 1e-13


def test_cross_rung_pair_receives_equal_and_opposite_kick():
    """The scheme's raison d'etre, in its smallest instance.

    Two particles on different rungs interact at level ``max(rung) = 1``. The
    *inactive* coarse partner must still receive the equal-and-opposite kick; if
    it does not, momentum leaks every sub-step.
    """
    positions = jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.2, -0.1]], dtype=jnp.float64)
    masses = jnp.asarray([1.0, 3.0], dtype=jnp.float64)
    rung = jnp.asarray([0, 1], dtype=jnp.int32)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=1, theta=0.6, max_order=4, leaf_size=1
    )
    fmm.prepare(positions, masses)

    level0 = fmm.level_accelerations(positions, masses, rung=rung, level=0)
    level1 = fmm.level_accelerations(positions, masses, rung=rung, level=1)
    # The only pair has max(rung) == 1, so level 0 is empty.
    assert float(jnp.linalg.norm(level0)) == 0.0
    forces = masses[:, None] * level1
    assert float(jnp.linalg.norm(forces[0] + forces[1])) < 1e-15 * float(
        jnp.linalg.norm(forces[0])
    )


@pytest.mark.parametrize("uniform_level", [0, 2, K_MAX])
def test_uniform_rung_reproduces_the_oracle_level_split(uniform_level):
    """With every particle on one rung the split is unambiguous, so it must match.

    No cell is rung-mixed, so the far field's cell-granularity assignment
    coincides with the per-particle predicate and the FMM must reproduce the
    dense oracle's decomposition exactly: all the force on ``uniform_level``, and
    identically nothing anywhere else. This pins the level bookkeeping without
    depending on the B2 approximation.
    """
    n = 1024
    positions, masses, _, state = _build(n, theta=0.9, order=6, leaf_size=16)
    rung = jnp.full((n,), uniform_level, dtype=jnp.int32)

    for level in range(K_MAX + 1):
        got = mutual_level_accelerations(
            state, positions, masses, rung=rung, level=level, k_max=K_MAX
        )
        if level != uniform_level:
            assert float(jnp.max(jnp.abs(got))) == 0.0
            continue
        want = _level_oracle(positions, masses, rung, level)
        error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert error < 1e-2


def test_cell_level_split_over_refines_but_still_partitions():
    """Document the B2 approximation: per-level differs, the sum does not.

    With rung-mixed cells the far field assigns a coarse particle the rung of the
    finest particle sharing its cell, so an individual level will *not* match the
    dense oracle -- that is the documented cost of keeping one multipole set per
    cell instead of ``k_max + 1``. What must still hold exactly is that the levels
    partition the total, which is what the integrator actually relies on.
    """
    n = 1024
    positions, masses, _, state = _build(n, theta=0.9, order=4, leaf_size=16)
    rung = _rungs(n)
    per_level = [
        mutual_level_accelerations(
            state, positions, masses, rung=rung, level=k, k_max=K_MAX
        )
        for k in range(K_MAX + 1)
    ]
    oracle_total = sum(
        _level_oracle(positions, masses, rung, k) for k in range(K_MAX + 1)
    )
    summed = sum(per_level)
    total_error = float(
        jnp.linalg.norm(summed - oracle_total) / jnp.linalg.norm(oracle_total)
    )
    assert total_error < 3e-2

    # At least one level must visibly disagree with the oracle, otherwise this
    # test is not actually exercising the rung-mixed regime it documents.
    disagreements = [
        float(
            jnp.linalg.norm(per_level[k] - _level_oracle(positions, masses, rung, k))
            / jnp.linalg.norm(_level_oracle(positions, masses, rung, k))
        )
        for k in range(K_MAX + 1)
    ]
    assert max(disagreements) > 1e-2


def test_fused_boundary_equals_the_weighted_sum_of_levels():
    """One traversal must equal the per-level sum it replaces.

    This is the correctness half of the fused-boundary primitive; the other half
    (that it really is one traversal) is structural -- ``boundary_kick`` makes a
    single call into the kernels.
    """
    positions, masses, _, state = _build(1024, theta=0.9, order=4, leaf_size=16)
    rung = _rungs(1024)
    dt_max = 0.05
    per_level = [
        mutual_level_accelerations(
            state, positions, masses, rung=rung, level=k, k_max=K_MAX
        )
        for k in range(K_MAX + 1)
    ]
    for s in range(n_sub(K_MAX) + 1):
        weights = boundary_level_weights(s, K_MAX, dt_max, dtype=jnp.float64)
        fused = mutual_weighted_accelerations(
            state, positions, masses, rung=rung, level_weights=weights
        )
        expected = sum(float(weights[k]) * per_level[k] for k in range(K_MAX + 1))
        error = float(
            jnp.linalg.norm(fused - expected) / (jnp.linalg.norm(expected) + 1e-300)
        )
        assert error < 1e-12, f"boundary {s}: {error}"
        assert _momentum_residual(fused, masses) < 1e-13


def test_boundary_kick_conserves_momentum():
    n = 1024
    positions, masses = _system(n)
    velocities = jnp.zeros_like(positions)
    rung = _rungs(n)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=K_MAX, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)
    for s in range(n_sub(K_MAX) + 1):
        kicked = fmm.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_level_floor(s, K_MAX),
            dt_max=0.05,
            half=0.5 if is_sync_boundary(s, K_MAX) else 1.0,
        )
        delta_p = jnp.sum(masses[:, None] * (kicked - velocities), axis=0)
        scale = jnp.sum(jnp.abs(masses[:, None] * (kicked - velocities)), axis=0)
        assert (
            float(jnp.linalg.norm(delta_p) / (jnp.linalg.norm(scale) + 1e-300)) < 1e-13
        )


# ---------------------------------------------------------------------------
# Phase 4 -- differentiability
# ---------------------------------------------------------------------------


def test_gradient_matches_direct_sum_gradient():
    """``grad(FMM)`` must match ``grad(direct sum)`` to the FMM's force accuracy.

    The strongest available check that the whole mutual pipeline -- symmetric
    scatter, dual M2L, L2L cascade, L2P -- transposes correctly.
    """
    positions, masses, topology, state = _build(512, theta=1.0, order=4, leaf_size=8)
    assert topology.num_far_pairs > 0

    def loss_fmm(p, m):
        return jnp.sum(mutual_accelerations(state, p, m) ** 2)

    def loss_direct(p, m):
        return jnp.sum(_direct_sum_accelerations(p, m) ** 2)

    got_p, got_m = jax.grad(loss_fmm, argnums=(0, 1))(positions, masses)
    want_p, want_m = jax.grad(loss_direct, argnums=(0, 1))(positions, masses)
    assert jnp.all(jnp.isfinite(got_p)) and jnp.all(jnp.isfinite(got_m))
    err_p = float(jnp.linalg.norm(got_p - want_p) / jnp.linalg.norm(want_p))
    err_m = float(jnp.linalg.norm(got_m - want_m) / jnp.linalg.norm(want_m))
    assert err_p < 5e-2
    assert err_m < 5e-2


def test_finite_difference_matches_autodiff():
    """FD-vs-AD on the mutual force itself, independent of any oracle."""
    positions, masses, topology, state = _build(256, theta=1.0, order=4, leaf_size=8)
    assert topology.num_far_pairs > 0
    direction = jnp.asarray(
        np.random.default_rng(5).normal(0.0, 1.0, positions.shape), dtype=jnp.float64
    )

    def loss(p):
        return jnp.sum(mutual_accelerations(state, p, masses) ** 2)

    analytic = float(jnp.vdot(jax.grad(loss)(positions), direction))
    h = 1e-6
    numeric = float(
        (loss(positions + h * direction) - loss(positions - h * direction)) / (2 * h)
    )
    assert abs(analytic - numeric) <= 1e-5 * max(abs(numeric), 1.0)


def test_level_accelerations_are_differentiable():
    positions, masses, _, state = _build(256, theta=1.0, order=4, leaf_size=8)
    rung = _rungs(256)

    def loss(p):
        return jnp.sum(
            mutual_level_accelerations(
                state, p, masses, rung=rung, level=2, k_max=K_MAX
            )
            ** 2
        )

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.linalg.norm(grad)) > 0.0


# ---------------------------------------------------------------------------
# Phase 4 -- trajectory
# ---------------------------------------------------------------------------


def _rollout(fmm, positions, velocities, masses, rung, *, dt_max, n_base):
    """Frozen-rung block-step rollout on the fused (one-traversal) path."""
    for _ in range(n_base):
        positions, velocities, _ = fmm.advance_base_step(
            positions, velocities, masses, rung=rung, dt_max=dt_max
        )
    return positions, velocities


def _total_energy(positions, velocities, masses, *, softening=SOFTENING, G=1.0):
    kinetic = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
    dr = positions[None, :, :] - positions[:, None, :]
    r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + softening**2)
    live = ~jnp.eye(positions.shape[0], dtype=bool)
    potential = (
        -0.5 * G * jnp.sum(jnp.where(live, masses[:, None] * masses[None, :] / r, 0.0))
    )
    return kinetic + potential


def test_block_step_rollout_conserves_momentum_and_bounds_energy():
    """End-to-end: momentum to round-off, energy bounded over many base steps."""
    n = 512
    positions, masses = _system(n, seed=3)
    velocities = jnp.asarray(
        np.random.default_rng(4).normal(0.0, 0.05, (n, 3)), dtype=jnp.float64
    )
    rung = _rungs(n, k_max=2, seed=6)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=2, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)

    p0 = jnp.sum(masses[:, None] * velocities, axis=0)
    e0 = _total_energy(positions, velocities, masses)
    final_x, final_v = _rollout(
        fmm, positions, velocities, masses, rung, dt_max=2.0e-3, n_base=6
    )
    p1 = jnp.sum(masses[:, None] * final_v, axis=0)
    e1 = _total_energy(final_x, final_v, masses)

    momentum_drift = float(jnp.linalg.norm(p1 - p0))
    scale = float(jnp.sum(jnp.abs(masses[:, None] * final_v)))
    assert momentum_drift / scale < 1e-13
    assert abs(float(e1 - e0) / float(abs(e0))) < 5e-3


def test_rollout_gradient_finite_difference():
    """FD-vs-AD of ``d(summary)/d(IC)`` through the multi-rung rollout.

    The trajectory case: the fixed-topology force composed across the base steps,
    with the rung schedule frozen so the map is globally smooth in the continuous
    state (nornax uses the same ``reassign_rungs=False`` setting for its own FD
    checks).
    """
    n = 128
    positions, masses = _system(n, seed=8)
    velocities = jnp.zeros_like(positions)
    rung = _rungs(n, k_max=1, seed=9)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=1, theta=1.0, max_order=4, leaf_size=8
    )
    topology_state = fmm.prepare(positions, masses)
    assert topology_state.topology.num_far_pairs > 0

    def summary(p):
        final_x, final_v = _rollout(
            fmm, p, velocities, masses, rung, dt_max=1.0e-3, n_base=2
        )
        return jnp.sum(final_x**2) + jnp.sum(final_v**2)

    direction = jnp.asarray(
        np.random.default_rng(10).normal(0.0, 1.0, positions.shape), dtype=jnp.float64
    )
    analytic = float(jnp.vdot(jax.grad(summary)(positions), direction))
    h = 1e-6
    numeric = float(
        (summary(positions + h * direction) - summary(positions - h * direction))
        / (2 * h)
    )
    assert abs(analytic - numeric) <= 1e-5 * max(abs(numeric), 1.0)


# ---------------------------------------------------------------------------
# Phase 5 -- backend selection
# ---------------------------------------------------------------------------


def test_pallas_backend_falls_back_without_a_supported_device():
    """The selector must degrade to pure JAX rather than fail off-GPU."""
    positions, masses, topology, jax_state = _build(
        512, theta=1.0, order=4, leaf_size=8
    )
    pallas_state = build_mutual_state(topology, softening=SOFTENING, use_pallas=True)
    assert topology.num_far_pairs > 0
    a_jax = mutual_accelerations(jax_state, positions, masses)
    a_pallas = mutual_accelerations(pallas_state, positions, masses)
    error = float(jnp.linalg.norm(a_pallas - a_jax) / jnp.linalg.norm(a_jax))
    assert error < 1e-10
    assert _momentum_residual(a_pallas, masses) < 1e-13


def test_pallas_m2l_kernel_matches_pure_jax_in_interpret_mode():
    """Actually run the Pallas z-M2L kernel, on CPU, via interpret mode.

    Without ``interpret`` this test would be vacuous off-GPU: ``use_pallas``
    silently falls back on CPU, so the assertion would compare pure JAX against
    itself. Interpret mode executes the real kernel logic, so a drifted Pallas
    z-M2L is caught by CPU CI -- the same trick the operator-level parity test
    uses.
    """
    positions, masses, topology, jax_state = _build(
        512, theta=1.0, order=4, leaf_size=8
    )
    assert topology.num_far_pairs > 0
    pallas_state = build_mutual_state(
        topology, softening=SOFTENING, use_pallas=True, pallas_interpret=True
    )
    a_jax = mutual_accelerations(jax_state, positions, masses)
    a_pallas = mutual_accelerations(pallas_state, positions, masses)
    error = float(jnp.linalg.norm(a_pallas - a_jax) / jnp.linalg.norm(a_jax))
    assert error < 1e-10
    assert _momentum_residual(a_pallas, masses) < 1e-13


def test_pallas_backend_is_differentiable():
    """``backend="pallas"`` must survive ``jax.grad``.

    ``pallas_call`` has no JVP/transpose, so a path that reaches the *bare* Pallas
    kernel is silently non-differentiable -- and on CPU the fallback hides it, so
    only interpret mode can catch the regression here. The mutual far field routes
    through jaccpot's ``custom_vjp`` wrapper instead, whose reverse is autodiff of
    the pure-JAX recurrence twin; the gradient must therefore match the pure-JAX
    path's.
    """
    positions, masses, topology, jax_state = _build(
        256, theta=1.0, order=4, leaf_size=8
    )
    assert topology.num_far_pairs > 0
    pallas_state = build_mutual_state(
        topology, softening=SOFTENING, use_pallas=True, pallas_interpret=True
    )

    def loss(state, p):
        return jnp.sum(mutual_accelerations(state, p, masses) ** 2)

    g_jax = jax.grad(lambda p: loss(jax_state, p))(positions)
    g_pallas = jax.grad(lambda p: loss(pallas_state, p))(positions)
    assert jnp.all(jnp.isfinite(g_pallas))
    error = float(jnp.linalg.norm(g_pallas - g_jax) / jnp.linalg.norm(g_jax))
    assert error < 1e-8


def test_pallas_near_field_kernel_matches_pure_jax_in_interpret_mode():
    """Run the mutual P2P kernel itself, on CPU, and compare the near field alone.

    Isolating the near field matters: at these sizes the far field dominates the
    *difference* between two total forces, so a broken near-field kernel could
    hide inside a total-force comparison. Here the far field is not evaluated at
    all.
    """
    from jaccpot.mutual.nearfield import mutual_near_field_forces

    positions, masses, topology, state = _build(512, theta=1.0, order=4, leaf_size=8)
    assert topology.num_near_pairs > 0
    fwd = state.forward_permutation
    pos_sorted, mass_sorted = positions[fwd], masses[fwd]

    def near(use_pallas, interpret):
        return mutual_near_field_forces(
            pos_sorted,
            mass_sorted,
            leaf_particles=state.leaf_particles,
            leaf_particle_valid=state.leaf_particle_valid,
            near_a=state.near_a,
            near_b=state.near_b,
            near_valid=state.near_valid,
            self_leaves=state.self_leaves,
            softening=SOFTENING,
            use_pallas=use_pallas,
            interpret=interpret,
        )

    f_jax = near(False, False)
    f_pallas = near(True, True)
    error = float(jnp.linalg.norm(f_pallas - f_jax) / jnp.linalg.norm(f_jax))
    assert error < 1e-10
    # The near field alone must already conserve momentum exactly -- it is a
    # closed set of equal-and-opposite pair impulses.
    total = jnp.sum(f_pallas, axis=0)
    scale = jnp.sum(jnp.abs(f_pallas), axis=0)
    assert float(jnp.linalg.norm(total) / jnp.linalg.norm(scale)) < 1e-13


def test_pallas_near_field_kernel_actually_runs_in_interpret_mode(monkeypatch):
    """Guard the parity test above against becoming vacuous.

    ``use_pallas`` degrades to pure JAX wherever the hardware cannot run the
    kernel, so a parity assertion can silently compare pure JAX against itself --
    which is exactly how a real bug survived on this branch once. Count the
    kernel invocations rather than trusting the flag.
    """
    import jaccpot.pallas.nearfield_mutual as nfm
    from jaccpot.mutual.nearfield import mutual_near_field_forces

    calls = []
    original = nfm.mutual_leafpair_block_pallas

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(nfm, "mutual_leafpair_block_pallas", spy)

    positions, masses, topology, state = _build(256, theta=1.0, order=4, leaf_size=8)
    fwd = state.forward_permutation
    mutual_near_field_forces(
        positions[fwd],
        masses[fwd],
        leaf_particles=state.leaf_particles,
        leaf_particle_valid=state.leaf_particle_valid,
        near_a=state.near_a,
        near_b=state.near_b,
        near_valid=state.near_valid,
        self_leaves=state.self_leaves,
        softening=SOFTENING,
        use_pallas=True,
        interpret=True,
    )
    # One launch for the intra-leaf blocks, one for the cross-leaf pairs.
    assert len(calls) >= 2


def test_pallas_near_field_analytic_reverse_matches_pure_jax():
    """The hand-written analytic reverse must equal the pure-JAX autodiff gradient.

    The mutual reverse differs from the gather-shaped rule in
    ``near_field.py::_leafpair_accel_analytic_vjp`` in one substantive way: each
    pair's cotangent is ``Fbar_a[i] - Fbar_b[j]``, fed from *both* endpoints,
    because the forward wrote to both. Dropping either term leaves a gradient
    that is wrong by roughly a factor of two on the cross terms and by nothing at
    all on a symmetric test function -- so the check is against a general
    (asymmetric) loss.
    """
    from jaccpot.mutual.nearfield import mutual_near_field_forces

    positions, masses, topology, state = _build(256, theta=1.0, order=4, leaf_size=8)
    fwd = state.forward_permutation
    mass_sorted = masses[fwd]
    weights = jnp.asarray(
        np.random.default_rng(3).normal(0.0, 1.0, (int(positions.shape[0]), 3))
    )

    def loss(p, use_pallas, interpret):
        forces = mutual_near_field_forces(
            p,
            mass_sorted,
            leaf_particles=state.leaf_particles,
            leaf_particle_valid=state.leaf_particle_valid,
            near_a=state.near_a,
            near_b=state.near_b,
            near_valid=state.near_valid,
            self_leaves=state.self_leaves,
            softening=SOFTENING,
            use_pallas=use_pallas,
            interpret=interpret,
        )
        return jnp.sum(forces * weights)

    pos_sorted = positions[fwd]
    g_jax = jax.grad(lambda p: loss(p, False, False))(pos_sorted)
    g_pallas = jax.grad(lambda p: loss(p, True, True))(pos_sorted)
    assert jnp.all(jnp.isfinite(g_pallas))
    error = float(jnp.linalg.norm(g_pallas - g_jax) / jnp.linalg.norm(g_jax))
    assert error < 1e-8


def test_pallas_near_field_respects_level_weights():
    """The level weight must be applied inside the kernel, symmetrically.

    A weight applied to only one side of the pair, or applied outside the kernel
    after the two reductions have already been rounded separately, breaks the
    exact cancellation. Both the value and the momentum residual are checked.
    """
    from jaccpot.mutual.nearfield import mutual_near_field_forces

    n = 512
    positions, masses, topology, state = _build(n, theta=1.0, order=4, leaf_size=8)
    fwd = state.forward_permutation
    rung = jnp.asarray(np.random.default_rng(4).integers(0, 4, n), dtype=jnp.int32)[fwd]
    level_weights = jnp.asarray([1.0, 0.5, 0.25, 0.125])

    def near(use_pallas, interpret):
        return mutual_near_field_forces(
            positions[fwd],
            masses[fwd],
            leaf_particles=state.leaf_particles,
            leaf_particle_valid=state.leaf_particle_valid,
            near_a=state.near_a,
            near_b=state.near_b,
            near_valid=state.near_valid,
            self_leaves=state.self_leaves,
            softening=SOFTENING,
            rung=rung,
            level_weights=level_weights,
            use_pallas=use_pallas,
            interpret=interpret,
        )

    f_jax = near(False, False)
    f_pallas = near(True, True)
    assert float(jnp.linalg.norm(f_pallas - f_jax) / jnp.linalg.norm(f_jax)) < 1e-10
    total = jnp.sum(f_pallas, axis=0)
    scale = jnp.sum(jnp.abs(f_pallas), axis=0)
    assert float(jnp.linalg.norm(total) / jnp.linalg.norm(scale)) < 1e-13


@pytest.mark.parametrize("order", [4, 6])
def test_far_field_chunk_padding_cannot_poison_the_expansion(monkeypatch, order):
    """Padding slots in the M2L scan must not produce a non-finite expansion.

    ``_dual_m2l`` pads its directed pair list out to a whole number of chunks with
    ``src == tgt == 0``, so a padding slot's ``delta`` is exactly zero and its
    radius is floored to 1e-30. The z-core then evaluates ``r**-(p+1)``, which at
    that floor is 5.6e151 at order 4 and 2.0e213 at order 6 — comfortably finite
    in float64, and far past ``inf`` in **float32**. The trailing
    ``* where(live, w, 0)`` then turns that ``inf`` into ``NaN`` rather than
    dropping it, poisoning the padded slot's target node, which the L2L cascade
    proceeds to broadcast across the whole tree.

    So this is a float32 defect at any practical order (float64 only overflows
    from order 10, where ``30 * (p + 1)`` passes 308), and it is reproduced here
    in float32 for that reason.

    It also needs padding to exist at all, which at the default 65536-pair budget
    takes > 65536 directed far pairs — larger than any test system. Shrinking the
    budget is what makes it reachable, and is exactly why the defect survived
    every existing size: at N=10⁴ the directed list is 17438 and fits one chunk,
    while at N=10⁵ it is 562392 and pads by 27432.
    """
    import jaccpot.mutual.farfield as farfield

    positions, masses = _system(512)
    positions = positions.astype(jnp.float32)
    masses = masses.astype(jnp.float32)
    topology, _ = build_mutual_topology(
        positions, masses, theta=1.0, order=order, leaf_size=8
    )
    assert topology.num_far_pairs > 0
    directed = 2 * topology.num_far_pairs
    # A budget one short of the list guarantees two chunks and a partial tail.
    monkeypatch.setattr(farfield, "_M2L_BATCH_BUDGET", max(1, directed - 1))

    state = build_mutual_state(topology, softening=SOFTENING)
    accelerations = mutual_accelerations(state, positions, masses)
    assert jnp.all(jnp.isfinite(accelerations))

    # The guard must be inert, not merely finite: the padded run has to agree with
    # the unpadded one, which never builds a degenerate slot in the first place.
    monkeypatch.setattr(farfield, "_M2L_BATCH_BUDGET", 1 << 16)
    reference = mutual_accelerations(
        build_mutual_state(topology, softening=SOFTENING), positions, masses
    )
    assert jnp.all(jnp.isfinite(reference))
    error = float(
        jnp.linalg.norm(accelerations - reference) / jnp.linalg.norm(reference)
    )
    assert error < 1e-5  # float32


def test_rung_above_k_max_is_rejected():
    """A rung with no kick weight is a configuration error, not a NaN later on."""
    positions, masses = _system(128)
    fmm = BlockStepFMM(softening=SOFTENING, k_max=1, leaf_size=16)
    fmm.prepare(positions, masses)
    rung = jnp.asarray(np.arange(128) % 4, dtype=jnp.int32)
    with pytest.raises(ValueError, match="k_max"):
        fmm.level_accelerations(positions, masses, rung=rung, level=1)


def test_unsupported_basis_and_backend_are_rejected():
    with pytest.raises(ValueError, match="basis"):
        BlockStepFMM(softening=SOFTENING, k_max=1, basis="cartesian")
    with pytest.raises(ValueError, match="backend"):
        BlockStepFMM(softening=SOFTENING, k_max=1, backend="cuda")


def test_force_call_under_tracing_without_prepare_raises():
    """A traced call cannot build a host topology; say so instead of crashing."""
    positions, masses = _system(64)
    fmm = BlockStepFMM(softening=SOFTENING, k_max=1, leaf_size=8)
    rung = jnp.zeros(64, dtype=jnp.int32)

    @jax.jit
    def run(p):
        return fmm.level_accelerations(p, masses, rung=rung, level=0)

    with pytest.raises(RuntimeError, match="prepare"):
        run(positions)
