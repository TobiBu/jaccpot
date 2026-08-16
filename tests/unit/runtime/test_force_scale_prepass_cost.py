"""Cost and reentrancy contracts for the adaptive force-scale prepass.

Dehnen's mass-dependent MAC (arXiv:1405.2255 eq (16a)) thresholds on ``eps *
min_b |a_b|`` over the target cell, so it needs a per-node force scale before the
traversal can run. ``mac_force_scale_mode="paper"`` obtains it from a full extra
low-order FMM solve on *every* ``prepare_state``, which is essentially all of the
measured 3.5x steady-state prepare overhead. Dehnen §5.4 licenses reusing the
previous step's accelerations instead ("only very slightly worse" than the exact
``a_b``), which is what ``"paper_cached"`` does.

These tests pin the three things that reuse has to get right: that the prepass
really only runs once, that reuse does not make ``prepare_state`` itself
history-dependent, and that the prepass -- an inner solve on the same particles --
does not corrupt the enclosing call's solver state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    RuntimePolicyConfig,
)
from tests.unit.runtime._reproducibility import assert_reproducible

# Compile-bound: every test here builds a solver and runs at least one full FMM
# solve, measured at 26-95 s each on CPU. `ci.yml` runs the version-compatibility
# matrix (`test-smoke`) with `-m "not slow and not experimental"` on a 30 minute
# budget and reserves the compile-heavy tests for `test-full` on 3.13. Leaving
# these unmarked put 94 such cases into that matrix and timed it out.
#
# `test_dehnen_mac_reference.py` is deliberately NOT marked: it checks eqs (12),
# (13), (15) and (16a) against independent numpy references at 1-10 s per test, so
# the criterion's correctness is still verified on every supported Python.
pytestmark = pytest.mark.slow


# eq (16a) is very conservative at small N: the threshold is set by the
# *least*-accelerated particle in the target cell, so N=64/leaf=4 accepts nothing
# at any eps in [1e-5, 1e-2]. N=512/leaf=8 at eps=3e-3 is the smallest setting
# that yields a non-empty M2L list, so all paper-mode cases below use it.
N_PARTICLES = 512
LEAF_SIZE = 8
MAX_ORDER = 4
PAPER_EPS = 3.0e-3
SOFTENING = 1.0e-3


def _sample_problem(n: int = N_PARTICLES):
    key_pos, key_mass = jax.random.split(jax.random.PRNGKey(20260731))
    positions = jax.random.normal(key_pos, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=jnp.float64)) + 0.5
    return positions, masses


def _traversal_cfg() -> RuntimePolicyConfig:
    return RuntimePolicyConfig(
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=131072,
            process_block=512,
            max_interactions_per_node=65536,
            max_neighbors_per_leaf=65536,
        )
    )


def _paper_solver(*, mode: str | None = None, **kwargs) -> FastMultipoleMethod:
    """Solver on the Dehnen paper MAC. ``mode=None`` keeps the shipped default."""

    extra = {} if mode is None else {"mac_force_scale_mode": mode}
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_eps=PAPER_EPS,
        advanced=FMMAdvancedConfig(
            mac_type="dehnen_error",
            runtime=_traversal_cfg(),
        ),
        **extra,
        **kwargs,
    )


def _count_paper_prepasses(monkeypatch, fmm: FastMultipoleMethod) -> list[int]:
    """Return a one-element list that counts paper-prepass invocations."""

    impl = fmm._impl
    calls = [0]
    original = type(impl)._compute_force_scale_paper_prepass_from_tree_artifacts

    def counting(self, **kwargs):
        calls[0] += 1
        return original(self, **kwargs)

    monkeypatch.setattr(
        type(impl),
        "_compute_force_scale_paper_prepass_from_tree_artifacts",
        counting,
    )
    return calls


def _prepare(fmm: FastMultipoleMethod, positions, masses):
    return fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )


# --------------------------------------------------------------------------- #
# paper_cached: the prepass runs once, not once per prepare_state
# --------------------------------------------------------------------------- #


def test_dehnen_error_default_mode_is_paper_cached():
    """The shipped default must not be the every-step prepass.

    ``mac_force_scale_mode`` defaults to ``"prev"``, which paper mode promotes.
    Promoting it to ``"paper"`` would charge every ``prepare_state`` a full extra
    FMM solve for a force scale §5.4 says may be reused.
    """

    assert _paper_solver()._impl.mac_force_scale_mode == "paper_cached"


def test_paper_cached_runs_the_prepass_only_on_the_cold_call(monkeypatch):
    positions, masses = _sample_problem()
    fmm = _paper_solver(mode="paper_cached")
    calls = _count_paper_prepasses(monkeypatch, fmm)

    for _ in range(3):
        state = _prepare(fmm, positions, masses)

    assert calls[0] == 1
    assert state.force_scale_nodes is not None


def test_paper_mode_runs_the_prepass_on_every_call(monkeypatch):
    """``"paper"`` is retained deliberately as the history-free upper bound."""

    positions, masses = _sample_problem()
    fmm = _paper_solver(mode="paper")
    calls = _count_paper_prepasses(monkeypatch, fmm)

    for _ in range(3):
        _prepare(fmm, positions, masses)

    assert calls[0] == 3


def test_paper_cached_cold_call_reproduces_paper_mode_exactly():
    """Reuse must not change *what* the cold call computes, only how often.

    Uses ``assert_reproducible`` rather than ``assert_array_equal`` because these
    are two INDEPENDENT solves, and the paper force scale is derived from a full
    FMM acceleration, which scatter-adds. On a GPU that lowers to atomics, so two
    runs of the same graph differ in the last bits -- see this directory's
    ``_reproducibility`` module, which was written for exactly this comparison.

    This became visible only once the GPU near-field auto policy moved off
    ``baseline``: the per-leaf-pair ``lax.scan`` it used to select accumulates one
    pair at a time in a fixed order, which happens to be bit-deterministic, while
    the bucketed traversal batches pairs and lets atomics order them. Measured
    drift here is 7.9e-16 relative, three orders inside the helper's 1e-13 band
    and many orders below the 23% mass loss and 15-41x error-tail blowups this
    area's real defects produced.
    """

    positions, masses = _sample_problem()

    cached = _paper_solver(mode="paper_cached")
    paper = _paper_solver(mode="paper")

    fs_cached = np.asarray(_prepare(cached, positions, masses).force_scale_nodes)
    fs_paper = np.asarray(_prepare(paper, positions, masses).force_scale_nodes)

    assert fs_cached.shape == fs_paper.shape
    assert_reproducible(fs_cached, fs_paper)


def test_paper_cached_rebuilds_the_scale_when_the_node_count_changes():
    """A cached scale is only reusable while it still indexes the same nodes."""

    positions, masses = _sample_problem()
    fmm = _paper_solver(mode="paper_cached")

    state_small = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    state_large = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE * 4, max_order=MAX_ORDER
    )

    n_small = int(state_small.force_scale_nodes.shape[0])
    n_large = int(state_large.force_scale_nodes.shape[0])
    assert n_small == int(state_small.tree.parent.shape[0])
    assert n_large == int(state_large.tree.parent.shape[0])
    assert n_small != n_large


# --------------------------------------------------------------------------- #
# reuse must not make prepare_state itself history-dependent
# --------------------------------------------------------------------------- #


def test_repeated_prepare_state_gives_bit_identical_accelerations():
    """Idempotency guard on the ``_last_force_scale_nodes`` statefulness.

    Two ``prepare_state`` calls on identical inputs must agree: the first writes
    the cache, the second reads it back, and reading it back must reproduce what
    writing it produced.

    The cached scale itself is compared to the bit -- that is the stateful thing
    under test, and it is exactly reproducible on every backend. The accelerations
    derived from it are compared within the atomic-reduction noise band, because
    on GPU the near-field/M2L scatter-adds commit in nondeterministic order and no
    repeat of *any* computation is bit-identical there. See
    ``_reproducibility.py`` for the measurement.
    """

    positions, masses = _sample_problem()
    fmm = _paper_solver(mode="paper_cached")

    first = _prepare(fmm, positions, masses)
    second = _prepare(fmm, positions, masses)

    np.testing.assert_array_equal(
        np.asarray(first.force_scale_nodes),
        np.asarray(second.force_scale_nodes),
    )
    assert_reproducible(
        np.asarray(fmm.evaluate_prepared_state(first)),
        np.asarray(fmm.evaluate_prepared_state(second)),
        err_msg="cache read-back changed the force beyond reduction noise",
    )


def test_paper_mode_is_history_free_across_repeated_solves():
    positions, masses = _sample_problem()
    fmm = _paper_solver(mode="paper")

    def solve():
        return np.asarray(
            fmm.compute_accelerations(
                positions,
                masses,
                leaf_size=LEAF_SIZE,
                max_order=MAX_ORDER,
                reuse_prepared_state=False,
            )
        )

    # History leaking between solves would move the force by far more than the
    # GPU's last-bit reduction noise -- a stale force scale shifts the error tail
    # by 15-41x when it bites at all.
    assert_reproducible(solve(), solve())


# --------------------------------------------------------------------------- #
# a stale cached scale has to be good enough -- Dehnen §5.4's actual claim
# --------------------------------------------------------------------------- #


def _dehnen_scaled_error(
    approx: np.ndarray, exact: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Dehnen's δa/f, on the cancellation-free force scale ``f_b``.

    ``f_b = sum_{a != b} G m_a / |x_a - x_b|^2``. The per-particle *relative* error
    δa/a is invalid here: wherever the vector sum ``a_b -> 0`` -- and the centre of
    a concentrated profile is exactly such a place -- it diverges for any MAC
    identically, so its tail measures the metric rather than the criterion.
    """

    delta = np.linalg.norm(approx - exact, axis=1)
    diff = positions[None, :, :] - positions[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    force_scale = np.sum(masses[None, :] / dist_sq, axis=1)
    return delta / force_scale


def _direct_accelerations(
    positions: np.ndarray, masses: np.ndarray, softening: float
) -> np.ndarray:
    diff = positions[None, :, :] - positions[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening * softening
    inv_cube = dist_sq ** (-1.5)
    np.fill_diagonal(inv_cube, 0.0)
    return np.einsum("ij,ij,ijk->ik", masses[None, :], inv_cube, diff)


# Per-step displacement, in units where the sample system has r_rms ~ 1.7. At this
# amplitude the two arms provably disagree about which pairs to accept (asserted
# below) and the error tail still holds, so the comparison is not vacuous. It is
# also far beyond any real timestep. The tail does *not* hold at every amplitude
# -- see the staleness cliff noted in docs/dehnen_mass_mac_status_and_plan.md.
STALENESS_DRIFT = 5.0e-2


def _far_pair_set(state) -> set[tuple[int, int]]:
    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    keep = (sources >= 0) & (targets >= 0)
    return set(zip(sources[keep].tolist(), targets[keep].tolist()))


def test_a_stale_cached_force_scale_does_not_degrade_the_error_tail():
    """The accuracy half of the trade: reuse must buy its speed-up for free.

    Dehnen §5.4 asserts the previous step's accelerations are "only very slightly
    worse" than the exact ``a_b`` as the eq (16a) force scale. Cost tests alone
    cannot see a regression here -- a cached scale that over-estimates
    ``min_b |a_b|`` loosens the threshold, so it over-accepts, so it is *faster*
    and wronger. This steps a small system forward and compares the error tail of
    ``paper_cached`` against ``paper`` on identical final positions, using Dehnen's
    own δa/f measure.
    """

    positions, masses = _sample_problem()
    drift = STALENESS_DRIFT * jax.random.normal(
        jax.random.PRNGKey(7), positions.shape, dtype=jnp.float64
    )

    def step_and_evaluate(mode: str):
        fmm = _paper_solver(mode=mode)
        for step in range(3):
            moved = positions + float(step) * drift
            state = _prepare(fmm, moved, masses)
            accelerations = fmm.evaluate_prepared_state(state)
        return np.asarray(accelerations), np.asarray(moved), _far_pair_set(state)

    acc_paper, final_positions, pairs_paper = step_and_evaluate("paper")
    acc_cached, cached_positions, pairs_cached = step_and_evaluate("paper_cached")
    np.testing.assert_array_equal(final_positions, cached_positions)

    # Non-vacuity: if the cached scale reproduced the fresh one's accept mask the
    # comparison below would hold for free and could not detect a regression.
    assert pairs_paper != pairs_cached

    masses_np = np.asarray(masses)
    exact = _direct_accelerations(final_positions, masses_np, SOFTENING)
    err_paper = _dehnen_scaled_error(acc_paper, exact, final_positions, masses_np)
    err_cached = _dehnen_scaled_error(acc_cached, exact, final_positions, masses_np)

    p99_paper = float(np.percentile(err_paper, 99))
    p99_cached = float(np.percentile(err_cached, 99))
    assert p99_paper > 0.0
    # "Only very slightly worse" -- allow 2x on the tail, which is far tighter
    # than the 1.2-1.9x tail *advantage* the criterion itself buys, so a genuine
    # accuracy regression from caching would still show up here.
    assert p99_cached <= 2.0 * p99_paper, (p99_cached, p99_paper)
    assert float(np.max(err_cached)) <= 4.0 * float(np.max(err_paper))


# --------------------------------------------------------------------------- #
# the prepass is an inner solve: it must not corrupt the outer call's state
# --------------------------------------------------------------------------- #


def _reuse_counts(fmm: FastMultipoleMethod, positions, masses, *, calls: int):
    counts = []
    for _ in range(calls):
        _prepare(fmm, positions, masses)
        entry = fmm._impl._topology_reuse_entry
        counts.append(None if entry is None else int(entry.reuse_count))
    return counts


def test_force_scale_prepass_does_not_consume_the_rebuild_every_budget():
    """The prepass must not halve the requested tree-rebuild cadence.

    The non-paper prepass re-enters ``prepare_state`` via
    ``compute_accelerations``, and the topology-reuse block increments
    ``reuse_count`` on the way through. Without a save/restore around the prepass
    each outer call spent two slots of the ``rebuild_every`` budget, so the tree
    was rebuilt twice as often as asked (the pre-fix sequence was 1, 3, 1, 3, ...
    for ``rebuild_every=4``) and ``recent_topology_reused`` reported a reuse even
    on calls that had just rebuilt.
    """

    positions, masses = _sample_problem(256)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_order=True,
        p_gears=(2, 3, 4),
        mac_force_scale_mode="prepass",
        reuse_topology=True,
        rebuild_every=4,
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )

    assert _reuse_counts(fmm, positions, masses, calls=6) == [0, 1, 2, 3, 0, 1]


@pytest.mark.parametrize("mode", ["paper", "paper_cached", "prepass", "prev"])
def test_a_prepass_never_requests_a_prepass_of_its_own(mode):
    """The prepass must not recurse into itself.

    The non-paper prepass re-enters ``prepare_state``, and ``"paper"`` set
    ``need_prepass`` unconditionally *ahead* of the reentrancy guard, so the inner
    call asked for another prepass, and so on until the interpreter stack ran out.
    Reachable straight from public kwargs, as configured here: ``adaptive_order``
    puts the solver on the force-scale path while ``tail_proxy`` keeps it off the
    paper traversal policy, which is the combination that takes the re-entrant
    ``compute_accelerations`` branch.
    """

    positions, masses = _sample_problem(128)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_order=True,
        p_gears=(2, 3),
        adaptive_error_model="tail_proxy",
        mac_force_scale_mode=mode,
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )

    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=3)

    scale = np.asarray(state.force_scale_nodes)
    assert scale.shape[0] == int(state.tree.parent.shape[0])
    assert np.all(np.isfinite(scale))


def test_force_scale_prepass_restores_the_outer_gear_bookkeeping():
    """The prepass runs a single gear; the outer call must still report its own.

    ``_recent_far_pairs_by_gear_counts`` is diagnostic output consumed by
    ``get_runtime_diagnostics``; leaking the prepass's one-gear tuple made the
    solver claim it had considered a single order.
    """

    positions, masses = _sample_problem(256)
    p_gears = (2, 3, 4)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_order=True,
        p_gears=p_gears,
        mac_force_scale_mode="prepass",
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )

    _prepare(fmm, positions, masses)

    assert fmm._impl.p_gears == p_gears
    assert len(fmm._impl._recent_far_pairs_by_gear_counts) == len(p_gears)


@pytest.mark.parametrize("mode", ["paper", "paper_cached", "prepass"])
def test_force_scale_prepass_restores_the_overridden_policy_knobs(mode):
    """Every knob the prepass overrides must be back to its configured value."""

    positions, masses = _sample_problem()
    fmm = _paper_solver(mode=mode)
    impl = fmm._impl
    before = (
        impl.p_gears,
        impl.adaptive_order,
        impl.adaptive_error_model,
        impl.mac_type,
    )

    _prepare(fmm, positions, masses)

    assert (
        impl.p_gears,
        impl.adaptive_order,
        impl.adaptive_error_model,
        impl.mac_type,
    ) == before
    assert impl._in_force_scale_prepass is False


# --------------------------------------------------------------------------- #
# "prev" now has a live writer, so it is no longer a silent no-op
# --------------------------------------------------------------------------- #


def _tail_proxy_solver(*, mode: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_order=True,
        p_gears=(2, 3, 4),
        adaptive_error_model="tail_proxy",
        mac_force_scale_mode=mode,
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )


def test_prev_mode_force_scale_is_populated_by_a_full_evaluation():
    """``"prev"`` was inert on the non-paper path: its only writer was dead code.

    ``_last_force_scale_nodes`` was written only inside the prepass, so a solver
    that never runs a prepass -- which is exactly what ``"prev"`` asks for -- fell
    through to a unit force scale on every step and stayed there. A unit scale is
    finite and non-None, so nothing caught it.
    """

    positions, masses = _sample_problem(256)
    fmm = _tail_proxy_solver(mode="prev")
    impl = fmm._impl

    assert impl._last_force_scale_nodes is None

    fmm.compute_accelerations(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    recorded = impl._last_force_scale_nodes
    assert recorded is not None
    recorded_np = np.asarray(recorded)
    assert np.all(np.isfinite(recorded_np))
    assert not np.allclose(recorded_np, 1.0)

    reused = _prepare(fmm, positions, masses)
    np.testing.assert_array_equal(
        np.asarray(reused.force_scale_nodes), recorded_np.astype(recorded_np.dtype)
    )


def test_target_subset_evaluation_does_not_write_the_force_scale_cache():
    """A target subset does not cover every node, so it must not seed the cache."""

    positions, masses = _sample_problem(256)
    fmm = _tail_proxy_solver(mode="prev")
    impl = fmm._impl

    state = _prepare(fmm, positions, masses)
    impl._last_force_scale_nodes = None
    fmm.evaluate_prepared_state(state, target_indices=jnp.arange(16, dtype=jnp.int32))

    assert impl._last_force_scale_nodes is None


def test_traced_evaluation_does_not_capture_a_tracer_in_solver_state():
    """The cache is an instance attribute; a tracer there would escape its trace.

    ``differentiable_accelerations`` does not route through
    ``evaluate_prepared_state``, so this guards the case where a caller traces the
    evaluation directly. Driven at the recorder to keep the assertion about the
    guard rather than about which surrounding paths happen to be traceable.
    """

    positions, masses = _sample_problem(256)
    fmm = _tail_proxy_solver(mode="prev")
    impl = fmm._impl
    state = _prepare(fmm, positions, masses)
    impl._last_force_scale_nodes = None
    acc_sorted = jnp.zeros(
        (int(state.tree.particle_indices.shape[0]), 3), dtype=jnp.float64
    )

    @jax.jit
    def record(acc):
        impl._record_force_scale_from_evaluation(
            state=state, evaluation=acc, full_evaluation=True
        )
        return acc

    record(acc_sorted)

    assert impl._last_force_scale_nodes is None
