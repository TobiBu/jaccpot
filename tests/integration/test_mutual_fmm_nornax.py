"""Cross-repo validation of :class:`jaccpot.nornax_adapter.BlockStepFMM`.

The library adapter must not import nornax -- the dependency graph is
``Jaccpot -> Yggdrax``, ``Nornax`` standalone, ``ODISSEO -> Nornax + Jaccpot`` --
but jaccpot's *tests* may, as a test-only dependency. So the structural claims
("it satisfies ``MutualForceModel``", "it reproduces the block schedule", "it
drives the real integrator") are checked here and nowhere else, and the whole
module skips when nornax is unavailable.

The oracle is ``nornax.forces.mutual_direct.MutualDirectSumGravity``: a dense
``O(N^2)`` sum whose momentum conservation is structural, so it is a fair
reference for the property under test and not just for the force values.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual import active_level_floor, is_sync_boundary, n_sub
from jaccpot.nornax_adapter import BlockStepFMM

# nornax fails to *import* (not merely to be found) against some diffrax
# releases -- `nornax.terms.NBodyTerm` is a non-frozen dataclass inheriting
# `diffrax.AbstractTerm`, which is frozen in diffrax < 0.7.2. Skip on any
# exception so an environment mismatch reports as "skipped", not "errored".
try:  # pragma: no cover - environment dependent
    from nornax.blockstep.rungs import assign_rungs
    from nornax.blockstep.schedule import (
        active_level_floor as nornax_active_level_floor,
    )
    from nornax.blockstep.schedule import is_sync_boundary as nornax_is_sync_boundary
    from nornax.blockstep.schedule import n_sub as nornax_n_sub
    from nornax.forces.base import MutualForceModel
    from nornax.forces.mutual_direct import MutualDirectSumGravity
    from nornax.solvers.leapfrog_kdk import (
        block_kdk_rollout,
        initialize_block_state,
        leapfrog_kdk_rollout,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"nornax unavailable: {exc!r}", allow_module_level=True)

SOFTENING = 1.0e-2


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(0.0, 1.0, (n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=jnp.float64)
    velocities = jnp.asarray(rng.normal(0.0, 0.05, (n, 3)), dtype=jnp.float64)
    return positions, velocities, masses


def _momentum(masses, vectors):
    return jnp.sum(masses[:, None] * vectors, axis=0)


def _relative_momentum(masses, vectors):
    terms = masses[:, None] * vectors
    scale = jnp.linalg.norm(jnp.sum(jnp.abs(terms), axis=0))
    return float(jnp.linalg.norm(jnp.sum(terms, axis=0)) / (float(scale) + 1e-300))


def test_adapter_satisfies_the_mutual_force_model_protocol():
    """The whole point of the structural design: no nornax import, still a match."""
    fmm = BlockStepFMM(softening=SOFTENING, k_max=2)
    assert isinstance(fmm, MutualForceModel)


def test_block_schedule_matches_nornax():
    """The duplicated schedule helpers must be bit-identical to nornax's.

    They are duplicated (not imported) to keep the adapter free of a nornax
    dependency, so this is the test that keeps the copy honest.
    """
    for k_max in range(5):
        assert n_sub(k_max) == nornax_n_sub(k_max)
        for s in range(n_sub(k_max) + 1):
            assert active_level_floor(s, k_max) == nornax_active_level_floor(s, k_max)
            assert is_sync_boundary(s, k_max) == nornax_is_sync_boundary(s, k_max)


def test_total_force_matches_the_direct_sum_oracle():
    """Summed over levels, the FMM must match the oracle to FMM tolerance.

    Only the *total* is compared: the far field splits at cell granularity, so an
    individual level is a different (equally valid) partition from the oracle's
    per-particle one. See :mod:`jaccpot.mutual.force`.
    """
    n, k_max = 1024, 2
    positions, _, masses = _system(n)
    rung = jnp.asarray(
        np.random.default_rng(1).integers(0, k_max + 1, n), dtype=jnp.int32
    )
    oracle = MutualDirectSumGravity(G=1.0, softening=SOFTENING)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=k_max, theta=0.7, max_order=6, leaf_size=16
    )
    fmm.prepare(positions, masses)

    reference = sum(
        oracle.level_accelerations(positions, masses, rung=rung, level=k)
        for k in range(k_max + 1)
    )
    got = sum(
        fmm.level_accelerations(positions, masses, rung=rung, level=k)
        for k in range(k_max + 1)
    )
    error = float(jnp.linalg.norm(got - reference) / jnp.linalg.norm(reference))
    assert error < 1e-2
    # The FMM's momentum residual must be no worse than the structurally-exact
    # oracle's -- that is the claim the whole mutual restructure exists to make.
    assert _relative_momentum(masses, got) < 1e-13
    assert _relative_momentum(masses, got) <= 1e3 * _relative_momentum(
        masses, reference
    )


def test_single_rung_leapfrog_rollout_conserves_momentum():
    """Phase-2 gate: drive nornax's textbook KDK with the full-source adapter."""
    n = 256
    positions, velocities, masses = _system(n, seed=2)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=0, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)

    state = initialize_block_state(positions, velocities, masses, fmm, k_max=0)
    p0 = _momentum(masses, state.velocities)
    final = leapfrog_kdk_rollout(state, 2.0e-3, fmm, n_steps=4)
    p1 = _momentum(masses, final.velocities)

    scale = float(jnp.sum(jnp.abs(masses[:, None] * final.velocities)))
    assert float(jnp.linalg.norm(p1 - p0)) / scale < 1e-13
    assert bool(jnp.all(jnp.isfinite(final.positions)))


def test_multi_rung_block_rollout_conserves_momentum():
    """Phase-3 gate: the real multi-rung block-step integrator on the FMM force."""
    n, k_max = 256, 1
    positions, velocities, masses = _system(n, seed=3)
    rung = jnp.asarray(
        np.random.default_rng(4).integers(0, k_max + 1, n), dtype=jnp.int32
    )
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=k_max, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)

    state = initialize_block_state(
        positions, velocities, masses, fmm, k_max=k_max, rung=rung
    )
    p0 = _momentum(masses, state.velocities)
    final = block_kdk_rollout(
        state,
        2.0e-3,
        fmm,
        k_max=k_max,
        n_base=2,
        checkpoint=False,
        reassign_rungs=False,
    )
    p1 = _momentum(masses, final.velocities)
    scale = float(jnp.sum(jnp.abs(masses[:, None] * final.velocities)))
    assert float(jnp.linalg.norm(p1 - p0)) / scale < 1e-13
    assert bool(jnp.all(jnp.isfinite(final.positions)))


def test_rung_assignment_round_trips_through_the_adapter():
    """nornax assigns rungs from the FMM's own acceleration; the loop must close."""
    n, k_max = 512, 2
    positions, _, masses = _system(n, seed=5)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=k_max, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)
    acc = fmm.total_accelerations(positions, masses)
    rung = assign_rungs(acc, dt_max=1.0e-2, k_max=k_max, eta=0.1, eps=SOFTENING)
    assert int(jnp.min(rung)) >= 0
    assert int(jnp.max(rung)) <= k_max
    # The assignment is severed from the gradient inside nornax; the adapter must
    # accept it unchanged.
    levels = [
        fmm.level_accelerations(positions, masses, rung=rung, level=k)
        for k in range(k_max + 1)
    ]
    total = sum(levels)
    error = float(jnp.linalg.norm(total - acc) / jnp.linalg.norm(acc))
    assert error < 1e-12


def test_fused_boundary_matches_the_per_level_integrator_path():
    """The fused primitive must reproduce nornax's per-level base step exactly.

    nornax's ``advance_base_step`` kicks level by level; the adapter's
    ``advance_base_step`` folds every active level into one traversal per
    boundary. Same arithmetic, fewer tree walks -- so the trajectories must agree
    to round-off, not merely to FMM tolerance.
    """
    from nornax.solvers.leapfrog_kdk import advance_base_step
    from nornax.state import BlockStepState

    n, k_max, dt_max = 256, 2, 1.0e-3
    positions, velocities, masses = _system(n, seed=6)
    rung = jnp.asarray(
        np.random.default_rng(7).integers(0, k_max + 1, n), dtype=jnp.int32
    )
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=k_max, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)

    state = BlockStepState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        acc=jnp.zeros_like(positions),
        rung=rung,
        base_index=jnp.asarray(0, dtype=jnp.int32),
    )
    per_level = advance_base_step(state, dt_max, fmm, k_max=k_max)
    fused_x, fused_v, _ = fmm.advance_base_step(
        positions, velocities, masses, rung=rung, dt_max=dt_max
    )
    assert (
        float(
            jnp.linalg.norm(fused_x - per_level.positions)
            / jnp.linalg.norm(per_level.positions)
        )
        < 1e-12
    )
    assert (
        float(
            jnp.linalg.norm(fused_v - per_level.velocities)
            / jnp.linalg.norm(per_level.velocities)
        )
        < 1e-10
    )
