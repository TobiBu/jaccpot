"""Differentiability of the force when the MAC itself depends on mass.

Dehnen (2014) eq (16a) makes the accept/reject decision a function of the source
cell's *mass* and multipole power, not only of geometry. jaccpot's fixed-topology
contract (docs/differentiable_fmm.md) puts that decision entirely on the frozen
side of the differentiated seam: ``prepare_state`` runs on the host, the policy
state is built there, and the accept/reject outcome is baked into
``state.interactions`` / ``state.neighbor_list`` before any tracing happens.

The consequence to pin here is that the mass-dependent MAC needs *no new
machinery* to be differentiable, and that the only thing it changes is where the
piecewise-constant boundary lies: ``d a / d m`` now inherits the same measure-zero
non-differentiable set that ``d a / d x`` already had.

Note on what each test can and cannot see:

- FD-vs-AD is a *self-consistency* check against the same frozen function. It is
  insensitive to whether the frozen topology is any good -- it will report ~1e-10
  agreement even if the criterion drops interactions or applies M2L where the
  multipole series diverges. It is necessary, not sufficient.
- grad(FMM) vs grad(direct sum) is the sensitive one, because it compares against
  an oracle that has no topology at all.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMAdvancedConfig
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
#
# One prepared state and ONE reverse pass serve every gradient test below (they all
# used the same solver, system and seed). Module fixtures hold arrays, which survive
# the per-test `jax.clear_caches()` conftest runs in this directory; the group keeps
# them to one xdist worker under CI's `-n 2 --dist loadgroup`.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("dehnen_mac_gradients")]


# eq (16a) thresholds on `eps * min_b |a_b|` over the *target* cell, so the least
# accelerated particle in a cell sets that cell's budget. At small N that makes the
# criterion very conservative: at N=64/leaf=4 it accepts nothing for any eps in
# [1e-5, 1e-2]. N=512/leaf=8 at eps=3e-3 is the smallest config measured to build a
# non-empty M2L list, which these tests need to cover the far-field reverse pass.
N_PARTICLES = 512
LEAF_SIZE = 8
MAX_ORDER = 4
EPS_MAC = 3.0e-3


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    probe = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    return positions, masses, probe


def _direct_sum_accelerations(positions, masses, *, softening, G):
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, dist2**-0.5, 0.0)
    weights = masses[None, :] * inv_dist**3
    weights = weights * (1.0 - jnp.eye(positions.shape[0], dtype=positions.dtype))
    return -G * jnp.einsum("ij,ijk->ik", weights, diffs)


def _num_far_pairs(state):
    inter = state.interactions
    if inter is not None:
        return int(jnp.sum(inter.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


def _mass_mac_solver(*, theta=1.0, eps=EPS_MAC, softening=1e-2, G=1.0, geometry="com"):
    cfg = FMMAdvancedConfig()
    advanced = replace(
        cfg,
        mac_type="dehnen_error",
        runtime=replace(
            cfg.runtime, retain_traversal_result=True, retain_interactions=True
        ),
    )
    return FastMultipoleMethod(
        basis="real",
        theta=theta,
        softening=softening,
        G=G,
        adaptive_eps=eps,
        dehnen_geometry_mode=geometry,
        advanced=advanced,
    )


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


_FD_STEP = 1e-6


@pytest.fixture(scope="module")
def mass_mac_problem():
    """The system, solver and frozen state every gradient test here differentiates.

    Returns
    -------
    tuple
        ``(fmm, state, positions, masses, probe)``.
    """
    positions, masses, probe = _system(N_PARTICLES, seed=1)
    fmm = _mass_mac_solver()
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    return fmm, state, positions, masses, probe


@pytest.fixture(scope="module")
def mass_mac_reverse(mass_mac_problem):
    """One reverse pass (both arguments), its FD references and the direct-sum oracle.

    The loss is jitted and then differentiated -- ``grad(jit(loss))`` -- so the same
    pass is the ``com`` case of the device-side geometry check below. The FD
    references perturb the same jitted function, so FD-vs-AD stays a
    self-consistency check on one frozen function.

    Parameters
    ----------
    mass_mac_problem : tuple
        Fixture value: solver, state, positions, masses, probe.

    Returns
    -------
    dict
        Gradients, directional finite differences, the force error and the
        direct-sum gradients.
    """
    fmm, state, positions, masses, probe = mass_mac_problem
    softening, G = 1e-2, 1.0  # `_mass_mac_solver`'s defaults

    @jax.jit
    def loss(pos, m):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, pos, m))

    def direct_loss(pos, m):
        return jnp.sum(
            probe * _direct_sum_accelerations(pos, m, softening=softening, G=G)
        )

    grad_positions, grad_masses = jax.grad(loss, argnums=(0, 1))(positions, masses)
    mass_direction = jnp.asarray(
        np.random.default_rng(2).normal(size=masses.shape), dtype=jnp.float64
    )
    step = _FD_STEP
    fd_positions = float(
        (loss(positions + step * probe, masses) - loss(positions - step * probe, masses))
        / (2 * step)
    )
    fd_masses = float(
        (
            loss(positions, masses + step * mass_direction)
            - loss(positions, masses - step * mass_direction)
        )
        / (2 * step)
    )
    force_error = _rel(
        fmm.differentiable_accelerations(state, positions, masses),
        _direct_sum_accelerations(positions, masses, softening=softening, G=G),
    )
    direct_positions, direct_masses = jax.grad(direct_loss, argnums=(0, 1))(
        positions, masses
    )
    return {
        "grad": {"positions": grad_positions, "masses": grad_masses},
        "direction": {"positions": probe, "masses": mass_direction},
        "fd": {"positions": fd_positions, "masses": fd_masses},
        "direct_grad": {"positions": direct_positions, "masses": direct_masses},
        "force_error": force_error,
    }


# ---------------------------------------------------------------------------
# FD vs AD on the frozen function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrt", ["positions", "masses"])
def test_fd_vs_ad_under_mass_dependent_mac(mass_mac_problem, mass_mac_reverse, wrt):
    """Reverse-mode AD must agree with FD of the same frozen-topology function.

    This holds trivially for a mass-dependent MAC because the MAC is frozen, so
    the test is a guard that nothing in the paper-policy path leaks a tracer into
    ``prepare_state`` or breaks the reverse pass -- not evidence that the
    criterion itself is sound.

    It also owns the ``f_b`` (``mac_force_scale_mode="paper_fb"``) case that
    ``test_fb_force_scale_estimator.py`` used to repeat: the force-scale prepass,
    whichever estimator, runs inside ``prepare_state`` and writes its caches there;
    ``differentiable_accelerations`` never records a force scale (only
    ``evaluate_prepared_state`` does), so under ``jax.grad`` the two modes run the
    same code on a different frozen state.
    """

    _, state, *_ = mass_mac_problem
    assert _num_far_pairs(state) > 0, "config must exercise the M2L reverse pass"
    grad = mass_mac_reverse["grad"][wrt]
    assert bool(jnp.all(jnp.isfinite(grad)))
    ad = float(jnp.sum(grad * mass_mac_reverse["direction"][wrt]))
    fd = mass_mac_reverse["fd"][wrt]
    assert abs(fd - ad) / (abs(fd) + 1e-300) < 1e-4


# ---------------------------------------------------------------------------
# grad(FMM) vs grad(direct sum) -- the sensitive check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrt", ["positions", "masses"])
def test_grad_matches_direct_sum_under_mass_dependent_mac(
    mass_mac_problem, mass_mac_reverse, wrt
):
    """The gradient must be as accurate as the force it differentiates.

    Bounded relative to the measured force error rather than to a constant: the
    ratio ``grad_error / force_error`` is the criterion-independent invariant, so
    this catches a criterion that silently degrades the force without having to
    re-tune a threshold whenever the MAC changes.
    """

    _, state, *_ = mass_mac_problem
    assert _num_far_pairs(state) > 0, "config must exercise the M2L reverse pass"
    force_error = mass_mac_reverse["force_error"]
    grad_error = _rel(
        mass_mac_reverse["grad"][wrt], mass_mac_reverse["direct_grad"][wrt]
    )

    assert force_error > 0.0, "far field must contribute a measurable error"
    assert grad_error < 3.0 * force_error + 1e-12, (
        f"grad error {grad_error:.3e} exceeds 3x the force error "
        f"{force_error:.3e}; the gradient is worse than the force it differentiates"
    )


# ---------------------------------------------------------------------------
# the measure-zero boundary claim
# ---------------------------------------------------------------------------


def _accept_signature(fmm, positions, masses) -> tuple[int, ...]:
    """A rebuilt topology fingerprint: the sorted accepted (target, source) pairs."""

    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    return _signature_of(state)


def _signature_of(state) -> tuple[int, ...]:
    """The sorted accepted (target, source) pairs of a prepared state."""

    inter = state.interactions
    assert inter is not None
    src = np.asarray(inter.sources)
    tgt = np.asarray(inter.targets)
    keep = (src >= 0) & (tgt >= 0)
    return tuple(sorted(zip(tgt[keep].tolist(), src[keep].tolist())))


def test_mass_perturbation_that_crosses_no_mac_boundary_is_smooth(
    mass_mac_problem, mass_mac_reverse
):
    """Off the boundary, the force is smooth in mass and AD is exact.

    "Crosses no boundary" is *verified* by rebuilding the accept decision at
    ``m +/- delta`` and requiring the pair set ``prepare_state`` built at ``m``,
    rather than assumed from a small delta.
    """

    fmm, state, positions, masses, probe = mass_mac_problem
    assert _num_far_pairs(state) > 0

    rng = np.random.default_rng(5)
    direction = jnp.asarray(rng.normal(size=masses.shape), dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    baseline = _signature_of(state)

    # Shrink delta until the rebuilt topology is provably unchanged both ways.
    delta = 1e-3
    for _ in range(30):
        if (
            _accept_signature(fmm, positions, masses + delta * direction) == baseline
            and _accept_signature(fmm, positions, masses - delta * direction)
            == baseline
        ):
            break
        delta *= 0.5
    else:  # pragma: no cover - would mean the boundary is dense in mass space
        pytest.fail("could not find a mass perturbation that crosses no MAC boundary")

    @jax.jit
    def loss(pos, m):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, pos, m))

    g0 = mass_mac_reverse["grad"]["masses"]
    g1 = jax.grad(loss, argnums=1)(positions, masses + delta * direction)
    assert bool(jnp.all(jnp.isfinite(g0))) and bool(jnp.all(jnp.isfinite(g1)))
    # Smoothness: the gradient moves by O(delta), not by O(1).
    drift = float(jnp.linalg.norm(g1 - g0) / (jnp.linalg.norm(g0) + 1e-300))
    assert drift < 100.0 * delta, (
        f"gradient drifted by {drift:.3e} for a mass step of {delta:.3e} that "
        "crosses no MAC boundary"
    )


def test_force_scale_nodes_receives_no_cotangent(mass_mac_problem):
    """``force_scale_nodes`` is a pytree leaf but must never carry a gradient.

    It is the p=1 prepass output that parameterises the frozen MAC decision. It
    is a leaf of ``FMMPreparedState``, so a future refactor could plausibly wire
    it into the differentiated seam -- where a cotangent would be meaningless,
    because the decision it drives is piecewise constant.
    """

    fmm, state, positions, masses, probe = mass_mac_problem
    if state.force_scale_nodes is None:
        pytest.skip("this configuration does not populate force_scale_nodes")

    def loss(force_scale):
        patched = replace(state, force_scale_nodes=force_scale)
        return jnp.sum(
            probe * fmm.differentiable_accelerations(patched, positions, masses)
        )

    grad = jax.grad(loss)(state.force_scale_nodes)
    assert float(jnp.max(jnp.abs(grad))) == 0.0

    # Stronger and metric-free: perturbing the frozen force scales must not move
    # the force, because the seam never reads them. A 1.5x perturbation that *was*
    # read would shift the accept mask and move the force by orders of magnitude,
    # so comparing within the GPU's atomic-reduction noise band still makes this
    # bite -- and on CPU it is still exact. See ``_reproducibility.py``.
    baseline = fmm.differentiable_accelerations(state, positions, masses)
    perturbed = fmm.differentiable_accelerations(
        replace(state, force_scale_nodes=state.force_scale_nodes * 1.5),
        positions,
        masses,
    )
    assert_reproducible(
        np.asarray(perturbed),
        np.asarray(baseline),
        err_msg="scaling force_scale_nodes moved the force: the seam read it",
    )


# ---------------------------------------------------------------------------
# traceability of the geometry modes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upward_artifacts():
    """A real tree, its sorted positions and the engine's upward sweep.

    Returns
    -------
    tuple
        ``(tree, positions_sorted, upward)`` for ``resolve_dehnen_geometry``.
    """
    from yggdrax.tree import Tree

    from jaccpot.runtime._fmm_impl import FMMEngine

    positions, masses, _ = _system(N_PARTICLES, seed=1)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=LEAF_SIZE,
        tree_type="radix",
        target_leaf_particles=LEAF_SIZE,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]
    masses_sorted = masses[tree.particle_indices]
    # The engine's real upward sweep (the module-level Cartesian one went in the
    # 2026-10 cleanup, K); the geometry modes read only centres and geometry.
    upward = FMMEngine(expansion_basis="solidfmm").prepare_upward_sweep(
        tree, positions_sorted, masses_sorted, max_order=MAX_ORDER, center_mode="com"
    )
    return tree, positions_sorted, upward


def _traced_geometry(mode, tree, upward):
    """``resolve_dehnen_geometry`` for ``mode`` as a function of traced positions."""

    from jaccpot.runtime._adaptive_policy import resolve_dehnen_geometry

    def run(pos):
        return resolve_dehnen_geometry(
            geometry_mode=mode,
            tree=tree,
            positions_sorted=pos,
            upward=upward,
            dtype=jnp.float64,
        )

    return run


def test_the_com_geometry_mode_is_differentiable(mass_mac_reverse):
    """The default (``com``) device-side mode must survive jit and grad.

    The fixture's reverse pass is ``grad(jit(loss))`` on a ``com``-geometry state,
    which is what this asserted with a reverse pass of its own.
    """

    for wrt in ("positions", "masses"):
        assert bool(jnp.all(jnp.isfinite(mass_mac_reverse["grad"][wrt]))), wrt


@pytest.mark.parametrize("mode", ["tree_approx", "runtime"])
def test_device_side_geometry_modes_build_a_usable_state(mode):
    """The opt-in device-side MAC geometry modes must build the frozen state.

    The mode is consumed in ``prepare_state``, outside any trace: the reverse pass
    differentiates the same frozen pytree whichever mode built it, and
    :func:`test_the_com_geometry_mode_is_differentiable` covers that pass. So these
    cases stop at the prepare; until 2026-10-12 each took a reverse pass of its
    own. (They are not traceable themselves, despite ``resolve_dehnen_geometry``'s
    error message: under ``jax.jit`` ``tree_approx`` raises a
    ``ConcretizationTypeError`` reading its leaf width off the device.)
    """

    positions, masses, _ = _system(N_PARTICLES, seed=1)
    state = _mass_mac_solver(geometry=mode).prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert _num_far_pairs(state) > 0, f"{mode}: no far pairs to differentiate"


@pytest.mark.parametrize("mode", ["exact", "tree"])
def test_host_loop_geometry_modes_reject_tracers_clearly(upward_artifacts, mode):
    """The numpy host-loop modes must fail with an actionable message.

    They run a Python loop over nodes, so they cannot be traced at all. Without an
    explicit guard the failure surfaces as an opaque numpy conversion error deep
    inside the policy builder.
    """

    tree, positions_sorted, upward = upward_artifacts
    run = _traced_geometry(mode, tree, upward)
    with pytest.raises(RuntimeError, match="cannot be traced"):
        jax.jit(lambda pos: run(pos)[1])(positions_sorted)
