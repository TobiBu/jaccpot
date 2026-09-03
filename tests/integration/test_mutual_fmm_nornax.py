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


# --- the explicit-topology hook (nornax block_kdk_rollout(rebuild_fn=...)) -----------
#
# nornax's rollout can carry a topology in its scan and rebuild it at base-step
# boundaries only, handing it to the force model as the explicit ``topology=``
# keyword. The adapter's half of that contract is ``rebuild_state`` (traceable,
# device backend) plus accepting the keyword on the three protocol methods. The
# tests below are the cross-repo half: the topology the model sees is frozen for a
# whole segment of ``rebuild_every`` base steps, and with ``rebuild_every = 1`` the
# rollout is the same map as ODISSEO's own scan-with-rebuild lane.


def _requires_topology_hook():
    """Skip when the installed nornax predates ``block_kdk_rollout(rebuild_fn=...)``."""
    import inspect

    if "rebuild_fn" not in inspect.signature(block_kdk_rollout).parameters:
        pytest.skip("installed nornax has no topology hook on block_kdk_rollout")


def _two_clumps(n=256, seed=25, separation=8.0, sigma=0.6):
    """Two well-separated Gaussian clumps, so the mutual MAC accepts far pairs.

    The single Gaussian blob ``_system`` builds has **no far pairs** at N = 256 for
    any ``theta`` this file uses (measured: 0 far, 120 near at ``theta = 0.9``), so
    on it the FMM is a direct sum and a "topology" claim is vacuous. Two clumps
    put a genuinely separated node pair in the tree at ODISSEO's production
    ``theta = 0.6``: 15 far pairs, 96 near, for this seed. Masses are unequal for
    the reason ODISSEO's own test bench gives: equal masses make a target-centric
    gather accidentally antisymmetric.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    centres = ([-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0])
    positions = np.concatenate([rng.normal(c, sigma, (half, 3)) for c in centres])
    velocities = rng.normal(0.0, 0.05, (n, 3))
    masses = rng.uniform(0.5, 1.5, n)
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(velocities, dtype=jnp.float64),
        jnp.asarray(masses, dtype=jnp.float64),
    )


def _device_fmm(k_max, *, theta=0.6, leaf_size=16):
    """A device-topology adapter: ``rebuild_state`` is traceable only on this backend."""
    return BlockStepFMM(
        softening=SOFTENING,
        k_max=k_max,
        theta=theta,
        max_order=4,
        leaf_size=leaf_size,
        topology_backend="device",
    )


def _topology_arrays(state):
    """The discrete structure of a device ``MutualFMMState``, as host arrays.

    ``fingerprint_topology`` (jaccpot.mutual.identity) digests a host-side
    ``MutualTopology``; the rollout carries the *device* state ``rebuild_state``
    returns, which has no ``MutualTopology`` behind it. So the comparison here is
    on the state's own index arrays -- the same six components the fingerprint
    covers, minus the tree shape, which the static-radix template fixes.
    """
    return {
        name: np.asarray(getattr(state, name))
        for name in (
            "forward_permutation",
            "inverse_permutation",
            "leaf_particles",
            "near_a",
            "near_b",
            "far_a",
            "far_b",
            "num_near_pairs",
            "num_far_pairs",
        )
    }


def _same_topology(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


class _RecordingTopology:
    """Delegate to an adapter and record, per traced call, which topology arrived.

    Recorded at trace time: within one traced base step every boundary kick and
    the end-of-step total must receive the *same* carried object, and none may
    receive ``None`` once the rollout carries one.
    """

    traced_boundary_weights = True

    def __init__(self, fmm):
        self._fmm = fmm
        self.k_max = fmm.k_max
        self.seen = []

    def level_accelerations(
        self, positions, masses, *, rung, level, args=None, topology=None
    ):
        self.seen.append(("level_accelerations", id(topology), topology is None))
        return self._fmm.level_accelerations(
            positions, masses, rung=rung, level=level, args=args, topology=topology
        )

    def total_accelerations(
        self, positions, masses, *, rung=None, args=None, topology=None
    ):
        self.seen.append(("total_accelerations", id(topology), topology is None))
        return self._fmm.total_accelerations(
            positions, masses, rung=rung, args=args, topology=topology
        )

    def boundary_kick(
        self,
        positions,
        velocities,
        masses,
        *,
        rung,
        level_weights=None,
        args=None,
        topology=None,
        **static,
    ):
        self.seen.append(("boundary_kick", id(topology), topology is None))
        return self._fmm.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            level_weights=level_weights,
            args=args,
            topology=topology,
            **static,
        )


def test_the_adapter_evaluates_against_an_explicit_topology():
    """``topology=`` selects the state; ``None`` falls back to the cached one.

    The explicit call must reproduce the cached call bit for bit when handed the
    very state ``prepare`` cached, on all three protocol methods, and a state
    built for a different particle count is refused rather than indexed.
    """
    n, k_max = 256, 2
    positions, velocities, masses = _system(n, seed=21)
    rung = jnp.asarray(
        np.random.default_rng(22).integers(0, k_max + 1, n), dtype=jnp.int32
    )
    fmm = _device_fmm(k_max)
    cached = fmm.prepare(positions, masses)
    weights = jnp.asarray([1.0e-3, 5.0e-4, 2.5e-4])

    for explicit, implicit in (
        (
            fmm.level_accelerations(
                positions, masses, rung=rung, level=1, topology=cached
            ),
            fmm.level_accelerations(positions, masses, rung=rung, level=1),
        ),
        (
            fmm.total_accelerations(positions, masses, topology=cached),
            fmm.total_accelerations(positions, masses),
        ),
        (
            fmm.boundary_kick(
                positions,
                velocities,
                masses,
                rung=rung,
                level_weights=weights,
                topology=cached,
            ),
            fmm.boundary_kick(
                positions, velocities, masses, rung=rung, level_weights=weights
            ),
        ),
    ):
        assert np.array_equal(np.asarray(explicit), np.asarray(implicit))

    # The explicit state is used, not merely accepted: a state rebuilt at displaced
    # positions gives a different (FMM-tolerance) answer for the same arguments.
    moved = fmm.rebuild_state(positions + 0.5, masses)
    a_cached = fmm.total_accelerations(positions, masses, topology=cached)
    a_moved = fmm.total_accelerations(positions, masses, topology=moved)
    assert not np.array_equal(np.asarray(a_cached), np.asarray(a_moved))

    with pytest.raises(ValueError, match="particles"):
        fmm.total_accelerations(positions[:-1], masses[:-1], topology=cached)


def test_the_frozen_topology_is_identical_across_a_segment_and_rebuilt_after_it():
    """Test 3 of the B4 plan: one topology per segment of ``rebuild_every`` base steps.

    With ``rebuild_every = 2`` the rollout is run for ``n_base = 1, 2, 3, 4``; the
    carried topology after 1 and 2 steps is the seed built at entry, bit for bit,
    and after 3 and 4 steps it is the one rebuilt before step 2 -- different from
    the seed, since the particles have moved. The rebuild count is taken at
    runtime with ``jax.debug.callback`` and must be ``ceil(n_base / 2)``. Inside a
    traced base step every fused call receives the same carried object, and none
    receives ``None``.
    """
    _requires_topology_hook()
    k_max, every, dt_max = 2, 2, 2.0e-2
    positions, velocities, masses = _two_clumps(seed=23)
    velocities = 10.0 * velocities  # move far enough per step to change pairs
    fmm = _device_fmm(k_max)
    seed = fmm.prepare(positions, masses)
    assert int(seed.num_far_pairs) > 0, "no far pairs: the topology claim is vacuous"
    recorder = _RecordingTopology(fmm)
    state = initialize_block_state(positions, velocities, masses, fmm, k_max=k_max)
    calls = []

    def rebuild_fn(p, m):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return fmm.rebuild_state(p, m)

    seen_topologies = {}
    for n_base in (1, 2, 3, 4):
        calls.clear()
        recorder.seen.clear()
        final = block_kdk_rollout(
            state,
            dt_max,
            recorder,
            k_max=k_max,
            n_base=n_base,
            eta=0.1,
            eps=SOFTENING,
            checkpoint=False,
            rebuild_fn=rebuild_fn,
            rebuild_every=every,
        )
        jax.block_until_ready(final)
        jax.effects_barrier()
        assert len(calls) == -(-n_base // every), n_base
        assert bool(jnp.all(jnp.isfinite(final.positions)))
        seen_topologies[n_base] = _topology_arrays(final.topology)
        # One traced base step, one object: the boundary kicks and the end-of-step
        # total all see the carried topology, never the cached fallback.
        assert recorder.seen, "the recorder saw no calls"
        assert not any(is_none for _, _, is_none in recorder.seen)
        assert len({obj for _, obj, _ in recorder.seen}) == 1
        assert {name for name, _, _ in recorder.seen} == {
            "boundary_kick",
            "total_accelerations",
        }

    assert _same_topology(seen_topologies[1], seen_topologies[2])
    assert _same_topology(seen_topologies[3], seen_topologies[4])
    assert not _same_topology(seen_topologies[2], seen_topologies[3])


def _odisseo_jitted_lane(
    fmm, positions, velocities, masses, *, dt_max, k_max, eta, eps, n_base
):
    """ODISSEO's ``integrate_blockstep_jitted`` base step, transcribed.

    ``odisseo/blockstep_coupling.py::integrate_blockstep_jitted`` (jaccpot-integration
    branch, d373b68) walks the boundaries itself because nornax had no place for a
    per-base-step topology. Its ``base_step`` is reproduced here verbatim in
    structure -- rebuild, full acceleration on the fresh topology, rung assignment,
    scanned boundary kicks off the weight table -- so that the rollout below has an
    oracle that does not depend on importing ODISSEO. ``freeze_template`` has been
    called by the caller, as ODISSEO does host-side before its scan.
    """
    steps = n_sub(k_max)
    dt_max = jnp.asarray(dt_max, dtype=positions.dtype)
    dt_min = dt_max / steps
    # nornax's dt_max-free table, as ODISSEO imports it (jaccpot's own takes dt_max).
    table = jnp.asarray(nornax_boundary_weight_table(k_max), dtype=positions.dtype)
    zero = jnp.asarray(0.0, dtype=positions.dtype)

    def base_step(carry, _):
        positions, velocities, mass = carry
        topology = fmm.rebuild_state(positions, mass)
        acc = fmm.weighted_accelerations(topology, positions, mass)
        rung = assign_rungs(acc, dt_max=dt_max, k_max=k_max, eta=eta, eps=eps)

        def boundary(inner, s):
            pos, vel = inner
            vel = vel + fmm.weighted_accelerations(
                topology, pos, mass, rung=rung, level_weights=dt_max * table[s]
            )
            pos = pos + jnp.where(s < steps, dt_min, zero) * vel
            return (pos, vel), None

        (positions, velocities), _ = jax.lax.scan(
            boundary, (positions, velocities), jnp.arange(steps + 1, dtype=jnp.int32)
        )
        return (positions, velocities, mass), jnp.bincount(rung, length=k_max + 1)

    (p, v, _), histograms = jax.jit(
        lambda p0, v0, m: jax.lax.scan(base_step, (p0, v0, m), xs=None, length=n_base)
    )(positions, velocities, masses)
    return p, v, histograms


def test_the_rollout_with_rebuild_every_one_matches_odisseos_jitted_lane():
    """Test 5 of the B4 plan: the hook reproduces ODISSEO's scan-with-rebuild lane.

    Same adapter, same template, same ``assign_rungs``, same weight table; the
    only structural difference is *which acceleration assigns the rungs*. ODISSEO
    evaluates it on the freshly rebuilt topology; nornax's base step reuses the
    end-of-step ``acc`` cached by the previous base step, which was evaluated on
    the previous topology at the same positions. The two differ at FMM tolerance,
    so a particle within that of a rung threshold could land on different rungs.
    The rung histograms are therefore asserted equal exactly (as ODISSEO's own
    lane-parity test does), and given equal rungs the kicks are the same
    ``weighted_accelerations`` calls on the same topology, so the trajectories
    must agree to round-off.

    Tolerance: 1e-12 relative on positions and velocities, and it was set after
    measuring, not to fit: against the *real* ``integrate_blockstep_jitted``
    (ODISSEO ``jaccpot-integration`` at d373b68, its own two-clump IC, N = 256,
    ``k_max = 2``, ``theta = 0.6``, four base steps, 2026-09-03) the two lanes
    were **bit-identical** in positions and velocities, with rung histogram
    ``[229, 23, 4]`` on both sides. 1e-12 is therefore not a fit to the observed
    difference (which is zero) but the level at which a disagreement would stop
    being round-off and start being a different topology or a different rung --
    FMM tolerance on this system is ~1e-4 -- with four orders of margin to spare.
    On this test's own two-clump system the transcribed oracle and the rollout
    were likewise bit-identical, with the schedule live: rung histogram
    ``[8, 234, 14]`` for three base steps and ``[8, 233, 15]`` at the fourth, on
    both sides.
    """
    _requires_topology_hook()
    # dt_max and eta were chosen by measuring the rung occupancy on this IC:
    # [8, 234, 14] over rungs 0..2, so all three levels are live and the two
    # lanes' rung assignments are genuinely compared rather than trivially equal.
    n, k_max, dt_max, eta, n_base = 256, 2, 2.5e-3, 0.15, 4
    positions, velocities, masses = _two_clumps(seed=25)
    fmm = _device_fmm(k_max)
    seed = fmm.prepare(positions, masses)  # freezes the template, as ODISSEO does
    assert int(seed.num_far_pairs) > 0, "no far pairs: the topology claim is vacuous"

    p_o, v_o, hist_o = _odisseo_jitted_lane(
        fmm,
        positions,
        velocities,
        masses,
        dt_max=dt_max,
        k_max=k_max,
        eta=eta,
        eps=SOFTENING,
        n_base=n_base,
    )

    state = initialize_block_state(positions, velocities, masses, fmm, k_max=k_max)
    final = jax.jit(
        lambda s: block_kdk_rollout(
            s,
            dt_max,
            fmm,
            k_max=k_max,
            n_base=n_base,
            eta=eta,
            eps=SOFTENING,
            checkpoint=False,
            rebuild_fn=fmm.rebuild_state,
            rebuild_every=1,
        )
    )(state)

    # The schedules must agree exactly; the final rung is the last one assigned.
    assert np.array_equal(
        np.asarray(jnp.bincount(final.rung, length=k_max + 1)), np.asarray(hist_o[-1])
    )
    assert (
        int(np.asarray(hist_o[-1]).max()) < n
    ), "the block scheme collapsed to one rung"
    assert (
        int(np.count_nonzero(np.asarray(hist_o[-1]))) == k_max + 1
    ), "a level is empty"

    for got, want, name in (
        (final.positions, p_o, "positions"),
        (final.velocities, v_o, "velocities"),
    ):
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < 1e-12, f"{name}: {rel:.3e}"


# --- nornax's conformance kit, run against the adapter ---------------------------------
#
# ``nornax.conformance.check_mutual_force_model`` is the runnable form of the
# ``MutualForceModel`` / ``FusedMutualForceModel`` contract (EDDA D-007): per-level
# momentum residual, levels partition the total, fused kick == per-level map at every
# boundary, explicit-topology parity, and the oracle match. The tests above check
# pieces of this by hand; this runs the whole kit, so a future contract addition
# lands here without a new hand-written test.


def _conformance_kit():
    """The kit, or a skip on a nornax that predates it."""
    try:
        from nornax.conformance import check_mutual_force_model
    except ImportError:  # pragma: no cover - depends on the installed nornax
        pytest.skip("installed nornax has no conformance kit")
    return check_mutual_force_model


@pytest.mark.parametrize("backend", ["host", "device"])
def test_the_adapter_passes_nornax_conformance_kit(backend):
    """Every check in nornax's kit, on both topology backends, non-vacuously.

    The system is the two-clump one with far pairs (asserted first: the kit
    cannot know whether an FMM configuration is a direct sum in disguise). The
    prepared state is passed as ``topology=``, so the explicit-topology contract
    from B4 is checked bit for bit against the implicit call. The oracle
    tolerance is set from the FMM's own accuracy at ``theta = 0.6``, order 4:
    measured 2.0e-5 relative L2 on this system (host backend, 15 far pairs),
    so 2e-3 is two orders of margin. Everything else is at the kit's round-off
    defaults; measured, the per-level momentum residuals were ~1e-17 and the
    fused-kick parities ~2.5e-16.
    """
    check = _conformance_kit()
    k_max = 2
    positions, velocities, masses = _two_clumps(seed=27)
    fmm = BlockStepFMM(
        softening=SOFTENING,
        k_max=k_max,
        theta=0.6,
        max_order=4,
        leaf_size=16,
        topology_backend=backend,
    )
    state = fmm.prepare(positions, masses)
    assert int(state.num_far_pairs) > 0, "no far pairs: the oracle check is vacuous"
    rung = jnp.asarray(
        np.random.default_rng(28).integers(0, k_max + 1, positions.shape[0]),
        dtype=jnp.int32,
    )

    report = check(
        fmm,
        positions,
        masses,
        k_max=k_max,
        rung=rung,
        dt_max=2.0e-3,
        topology=state,
        oracle=MutualDirectSumGravity(G=1.0, softening=SOFTENING),
        oracle_tolerance=2.0e-3,
        require_fused=True,
    )
    assert report.passed, report.summary()
    assert report.fused_selected and report.scanned_boundaries
    # The kit ran the checks this backend is about, not a subset of them.
    names = {c.name for c in report.checks}
    assert {
        "partition",
        "total_accelerations",
        "oracle",
        "explicit topology: total",
    } <= names
    assert sum(name.endswith(": kick") for name in names) == 2**k_max + 1


def test_the_carried_topology_can_be_checked_for_overflow_after_the_rollout():
    """The overflow check the B4 hook allows: on ``final.topology``, after the scan.

    ``raise_on_overflow`` cannot fire under the trace, so a driver that rebuilds
    inside ``block_kdk_rollout`` checks the carried topology afterwards -- the
    pattern ODISSEO's deletion (B5 step 2) will use for its last base step. On a
    healthy system it is silent and the far field is exercised.
    """
    _requires_topology_hook()
    from jaccpot.nornax_adapter import assert_far_field_is_exercised, raise_on_overflow

    k_max = 2
    positions, velocities, masses = _two_clumps(seed=29)
    fmm = _device_fmm(k_max)
    fmm.prepare(positions, masses)
    assert assert_far_field_is_exercised(fmm) > 0
    state = initialize_block_state(positions, velocities, masses, fmm, k_max=k_max)
    final = block_kdk_rollout(
        state,
        2.0e-3,
        fmm,
        k_max=k_max,
        n_base=2,
        eta=0.1,
        eps=SOFTENING,
        checkpoint=False,
        rebuild_fn=fmm.rebuild_state,
        rebuild_every=1,
    )
    raise_on_overflow(final.topology, fmm)
    assert assert_far_field_is_exercised(final.topology) > 0
