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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual import (
    active_level_floor,
    boundary_weight_table,
    is_sync_boundary,
    n_sub,
)
from jaccpot.nornax_adapter import BlockStepFMM

# nornax is a test-only dependency, so skip cleanly when it is absent. The
# `except Exception` (rather than ImportError) is deliberate: nornax used to fail
# to *import* against equinox < 0.13 / diffrax < 0.7.2, because
# `nornax.terms.NBodyTerm` was a non-frozen dataclass inheriting a frozen
# `AbstractTerm`. That is fixed upstream and no version overlay is needed any
# more, but an import-time TypeError should still report as "skipped", not
# "errored".
try:  # pragma: no cover - environment dependent
    from nornax.blockstep.rungs import assign_rungs
    from nornax.blockstep.schedule import (
        active_level_floor as nornax_active_level_floor,
    )
    from nornax.blockstep.schedule import (
        boundary_weight_table as nornax_boundary_weight_table,
    )
    from nornax.blockstep.schedule import is_sync_boundary as nornax_is_sync_boundary
    from nornax.blockstep.schedule import n_sub as nornax_n_sub
    from nornax.forces.base import FusedMutualForceModel, MutualForceModel
    from nornax.forces.mutual_direct import MutualDirectSumGravity
    from nornax.solvers.leapfrog_kdk import (
        advance_base_step,
        block_kdk_rollout,
        fused_boundary_model,
        initialize_block_state,
        leapfrog_kdk_rollout,
        supports_traced_level_weights,
    )
    from nornax.state import BlockStepState
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


def test_adapter_satisfies_the_mutual_force_model_protocols():
    """The whole point of the structural design: no nornax import, still a match.

    Both protocols, with no adapter changes: ``MutualForceModel`` for the per-level
    contract and ``FusedMutualForceModel`` for the per-boundary one. The fused
    protocol is separate precisely because ``MutualForceModel`` is
    ``runtime_checkable`` -- widening it would have made ``isinstance`` reject
    every existing implementation.
    """
    fmm = BlockStepFMM(softening=SOFTENING, k_max=2)
    assert isinstance(fmm, MutualForceModel)
    assert isinstance(fmm, FusedMutualForceModel)


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


def test_boundary_weight_table_matches_nornax():
    """The two repos' weight tables must be the same numbers, bit for bit.

    Both sides build the table independently -- jaccpot's takes ``dt_max``
    directly, nornax's is ``dt_max``-free and scaled by the integrator -- so this
    is the test that keeps the duplication honest. It is also the reason the split
    is safe: ``half`` and ``1 / 2**k`` are powers of two, so scaling only shifts an
    exponent and ``dt_max * (half / 2**k)`` lands on the same float as
    ``half * dt_max / 2**k``.
    """
    dt_max = 0.0125
    for k_max in range(5):
        ours = boundary_weight_table(k_max, dt_max, dtype=jnp.float64)
        theirs = dt_max * jnp.asarray(
            nornax_boundary_weight_table(k_max), dtype=jnp.float64
        )
        assert ours.shape == theirs.shape == (n_sub(k_max) + 1, k_max + 1)
        assert jnp.array_equal(ours, theirs), f"k_max={k_max}"


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
    """Phase-3 gate: the real multi-rung block-step integrator on the FMM force.

    This now exercises nornax's *fused* path -- and, since the adapter takes a
    traced ``level_weights`` vector, its scanned form: the adapter satisfies
    ``FusedMutualForceModel`` and the integrator opts in automatically. The
    assertion below pins that: if fusion silently stopped being selected the
    rollout would still pass, just at one traversal per active level instead of
    one per boundary.
    """
    n, k_max = 256, 1
    positions, velocities, masses = _system(n, seed=3)
    rung = jnp.asarray(
        np.random.default_rng(4).integers(0, k_max + 1, n), dtype=jnp.int32
    )
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=k_max, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)
    assert fused_boundary_model(fmm, k_max) is fmm

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


class _PerLevelOnly:
    """Expose only ``level_accelerations``, to force nornax's per-level path.

    ``advance_base_step`` probes for fusion with
    ``isinstance(force, FusedMutualForceModel)``, and ``BlockStepFMM`` satisfies
    it -- so passing the adapter directly now takes the *fused* path. Comparing
    the two paths therefore needs a model that deliberately fails the probe. It
    delegates to the same FMM, so the two runs differ only in how the integrator
    drives it.
    """

    def __init__(self, fmm):
        self._fmm = fmm

    def level_accelerations(self, positions, masses, *, rung, level, args=None):
        return self._fmm.level_accelerations(
            positions, masses, rung=rung, level=level, args=args
        )


class _StaticWeightsOnly:
    """Expose the fused primitive in its *static* form only, so nornax unrolls.

    nornax probes a fused backend for traced ``level_weights`` support
    (``supports_traced_level_weights``) and, when it is there, walks the sub-step
    boundaries with a ``lax.scan`` over a weight table instead of unrolling them.
    ``BlockStepFMM`` has it, so comparing the two integrator paths needs a model
    that deliberately fails the probe: this wrapper's ``boundary_kick`` has no
    ``level_weights`` parameter at all -- the original cross-repo contract, and
    what a backend written before the traced-weight seam looks like. It delegates
    to the same FMM, so the two runs differ only in how nornax drives it.
    """

    def __init__(self, fmm):
        self._fmm = fmm
        self.k_max = fmm.k_max

    def level_accelerations(self, positions, masses, *, rung, level, args=None):
        return self._fmm.level_accelerations(
            positions, masses, rung=rung, level=level, args=args
        )

    def total_accelerations(self, positions, masses, *, rung=None, args=None):
        return self._fmm.total_accelerations(positions, masses, rung=rung, args=args)

    def boundary_kick(
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        active_floor,
        dt_max,
        half=1.0,
        args=None,
    ):
        return self._fmm.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_floor,
            dt_max=dt_max,
            half=half,
            args=args,
        )


def test_adapter_advertises_traced_level_weights_to_nornax():
    """nornax must detect that this backend takes a traced weight vector.

    That detection is what decides whether nornax scans the boundaries (one traced
    boundary kick for the whole base step) or unrolls them (``2**k_max`` of them,
    each a full traversal's worth of graph for an FMM). It works off
    :meth:`BlockStepFMM.boundary_kick`'s signature, so renaming or dropping the
    ``level_weights`` parameter would silently cost the trace-size win -- this is
    the test that would catch it.
    """
    fmm = BlockStepFMM(softening=SOFTENING, k_max=2)
    assert supports_traced_level_weights(fmm)
    # And the static-only spelling of the same backend must decline it, or the
    # comparison in the parity test below would not compare two paths.
    assert not supports_traced_level_weights(_StaticWeightsOnly(fmm))
    assert fused_boundary_model(_StaticWeightsOnly(fmm), 2) is not None


def test_nornax_scanned_and_unrolled_fused_paths_agree_on_the_fmm_force():
    """Scanning the boundaries changes the graph, not the trajectory -- on the real FMM.

    nornax's own suite pins this against its direct sum; here the same claim is
    checked with the backend the primitive exists for, where each boundary is a
    genuine mutual traversal. Both runs use nornax's ``advance_base_step`` and the
    same prepared tree: one drives the FMM through ``level_weights=table[s]`` from
    inside a ``lax.scan``, the other through static ``active_floor``/``half`` in an
    unrolled Python loop. Agreement is to round-off (the weights are bit-identical;
    the two graph shapes let XLA associate and contract differently).
    """
    n, k_max, dt_max = 256, 2, 1.0e-3
    positions, velocities, masses = _system(n, seed=10)
    rung = jnp.asarray(
        np.random.default_rng(11).integers(0, k_max + 1, n), dtype=jnp.int32
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

    scanned = advance_base_step(state, dt_max, fmm, k_max=k_max)
    # The scanned path inlines the whole traversal into one scan program, where the
    # unrolled one reuses the inner kernels' cached executables -- so this test
    # compiles two distinct programs, and holding the first while building the
    # second is the memory pattern that put ``test_scanned_base_step_matches_the
    # _unrolled_one`` near the CI worker's ceiling. Drop the first before the
    # second, as that test does: measured peak RSS 1.64 -> 1.46 GB.
    jax.clear_caches()
    unrolled = advance_base_step(state, dt_max, _StaticWeightsOnly(fmm), k_max=k_max)

    for got, want, name in (
        (scanned.positions, unrolled.positions, "positions"),
        (scanned.velocities, unrolled.velocities, "velocities"),
        (scanned.acc, unrolled.acc, "acc"),
    ):
        error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert error < 1e-13, f"{name}: {error}"

    # Neither path may erode momentum: the weights multiply an already-mutual
    # +f/-f pair force, so this holds however the boundaries are walked.
    for label, result in (("scanned", scanned), ("unrolled", unrolled)):
        drift = _momentum(masses, result.velocities) - _momentum(masses, velocities)
        scale = float(jnp.sum(jnp.abs(masses[:, None] * result.velocities)))
        assert float(jnp.linalg.norm(drift)) / scale < 1e-13, label


def test_adapter_is_selected_for_the_fused_path():
    """nornax must actually *choose* fusion for this backend, not just tolerate it.

    ``fused_boundary_model`` is the opt-in gate: it requires the
    ``FusedMutualForceModel`` protocol *and* a ``k_max`` matching the integrator's.
    If either drifted, the integrator would silently fall back to the per-level
    path -- correct, but paying one traversal per active level, which is the whole
    cost the fused primitive exists to avoid.
    """
    fmm = BlockStepFMM(softening=SOFTENING, k_max=2)
    assert isinstance(fmm, FusedMutualForceModel)
    assert fused_boundary_model(fmm, 2) is fmm
    # A model whose k_max disagrees is a misconfiguration, not a fallback: its
    # fused weights would span the wrong level range.
    with pytest.raises(ValueError, match="k_max"):
        fused_boundary_model(fmm, 3)
    # And a per-level-only model must decline fusion rather than crash.
    assert fused_boundary_model(_PerLevelOnly(fmm), 2) is None


def test_nornax_fused_path_matches_its_per_level_path_on_the_fmm_force():
    """The core cross-repo claim: fusing changes cost, not the trajectory.

    Both runs use nornax's own ``advance_base_step`` and the same FMM force; only
    the code path differs -- ``n_sub + 2`` fused evaluations against
    ``sum_s (active levels at s)`` per-level ones. Agreement is to round-off
    rather than bit-for-bit, because the per-level path's ``half`` is a traced
    ``where`` while the fused path carries it baked into its weight table.
    """
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

    fused = advance_base_step(state, dt_max, fmm, k_max=k_max)
    per_level = advance_base_step(state, dt_max, _PerLevelOnly(fmm), k_max=k_max)

    for got, want, name in (
        (fused.positions, per_level.positions, "positions"),
        (fused.velocities, per_level.velocities, "velocities"),
        (fused.acc, per_level.acc, "acc"),
    ):
        error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert error < 1e-10, f"{name}: {error}"

    # Both paths must conserve momentum; fusion cannot be allowed to erode it.
    for label, result in (("fused", fused), ("per-level", per_level)):
        drift = _momentum(masses, result.velocities) - _momentum(masses, velocities)
        scale = float(jnp.sum(jnp.abs(masses[:, None] * result.velocities)))
        assert float(jnp.linalg.norm(drift)) / scale < 1e-13, label


def test_jaccpot_base_step_matches_nornax_fused_base_step():
    """The adapter's own base step and nornax's fused one must be the same map.

    The adapter provides `advance_base_step` so jaccpot can be driven without
    nornax at all; this keeps the two implementations of the same palindrome from
    drifting apart.
    """
    n, k_max, dt_max = 256, 2, 1.0e-3
    positions, velocities, masses = _system(n, seed=8)
    rung = jnp.asarray(
        np.random.default_rng(9).integers(0, k_max + 1, n), dtype=jnp.int32
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
    nornax_result = advance_base_step(state, dt_max, fmm, k_max=k_max)
    x, v, acc = fmm.advance_base_step(
        positions, velocities, masses, rung=rung, dt_max=dt_max
    )
    for got, want, name in (
        (x, nornax_result.positions, "positions"),
        (v, nornax_result.velocities, "velocities"),
        (acc, nornax_result.acc, "acc"),
    ):
        error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert error < 1e-10, f"{name}: {error}"
