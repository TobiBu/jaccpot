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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMAdvancedConfig

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
        runtime=replace(cfg.runtime, retain_traversal_result=True,
                        retain_interactions=True),
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


# ---------------------------------------------------------------------------
# FD vs AD on the frozen function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrt", ["positions", "masses"])
def test_fd_vs_ad_under_mass_dependent_mac(wrt):
    """Reverse-mode AD must agree with FD of the same frozen-topology function.

    This holds trivially for a mass-dependent MAC because the MAC is frozen, so
    the test is a guard that nothing in the paper-policy path leaks a tracer into
    ``prepare_state`` or breaks the reverse pass -- not evidence that the
    criterion itself is sound.
    """

    positions, masses, probe = _system(N_PARTICLES, seed=1)
    fmm = _mass_mac_solver()
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    assert _num_far_pairs(state) > 0, "config must exercise the M2L reverse pass"

    if wrt == "positions":
        def loss(x):
            return jnp.sum(
                probe * fmm.differentiable_accelerations(state, x, masses)
            )
        x0, direction = positions, probe
    else:
        def loss(m):
            return jnp.sum(
                probe * fmm.differentiable_accelerations(state, positions, m)
            )
        x0 = masses
        direction = jnp.asarray(
            np.random.default_rng(2).normal(size=masses.shape), dtype=jnp.float64
        )

    grad = jax.grad(loss)(x0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    step = 1e-6
    ad = float(jnp.sum(grad * direction))
    fd = float((loss(x0 + step * direction) - loss(x0 - step * direction)) / (2 * step))
    assert abs(fd - ad) / (abs(fd) + 1e-300) < 1e-4


# ---------------------------------------------------------------------------
# grad(FMM) vs grad(direct sum) -- the sensitive check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrt", ["positions", "masses"])
def test_grad_matches_direct_sum_under_mass_dependent_mac(wrt):
    """The gradient must be as accurate as the force it differentiates.

    Bounded relative to the measured force error rather than to a constant: the
    ratio ``grad_error / force_error`` is the criterion-independent invariant, so
    this catches a criterion that silently degrades the force without having to
    re-tune a threshold whenever the MAC changes.
    """

    positions, masses, probe = _system(N_PARTICLES, seed=1)
    softening, G = 1e-2, 1.0
    fmm = _mass_mac_solver(softening=softening, G=G)
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    assert _num_far_pairs(state) > 0, "config must exercise the M2L reverse pass"

    def fmm_loss(pos, m):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, pos, m))

    def direct_loss(pos, m):
        return jnp.sum(
            probe * _direct_sum_accelerations(pos, m, softening=softening, G=G)
        )

    force_error = _rel(
        fmm.differentiable_accelerations(state, positions, masses),
        _direct_sum_accelerations(positions, masses, softening=softening, G=G),
    )

    argnum = 0 if wrt == "positions" else 1
    g_fmm = jax.grad(fmm_loss, argnums=argnum)(positions, masses)
    g_direct = jax.grad(direct_loss, argnums=argnum)(positions, masses)
    grad_error = _rel(g_fmm, g_direct)

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

    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    inter = state.interactions
    assert inter is not None
    src = np.asarray(inter.sources)
    tgt = np.asarray(inter.targets)
    keep = (src >= 0) & (tgt >= 0)
    return tuple(sorted(zip(tgt[keep].tolist(), src[keep].tolist())))


def test_mass_perturbation_that_crosses_no_mac_boundary_is_smooth():
    """Off the boundary, the force is smooth in mass and AD is exact.

    "Crosses no boundary" is *verified* by rebuilding the accept decision at
    ``m`` and at ``m +/- delta`` and requiring an identical pair set, rather than
    assumed from a small delta.
    """

    positions, masses, probe = _system(N_PARTICLES, seed=1)
    fmm = _mass_mac_solver()
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    assert _num_far_pairs(state) > 0

    rng = np.random.default_rng(5)
    direction = jnp.asarray(rng.normal(size=masses.shape), dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    baseline = _accept_signature(fmm, positions, masses)

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

    def loss(m):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, positions, m))

    g0 = jax.grad(loss)(masses)
    g1 = jax.grad(loss)(masses + delta * direction)
    assert bool(jnp.all(jnp.isfinite(g0))) and bool(jnp.all(jnp.isfinite(g1)))
    # Smoothness: the gradient moves by O(delta), not by O(1).
    drift = float(jnp.linalg.norm(g1 - g0) / (jnp.linalg.norm(g0) + 1e-300))
    assert drift < 100.0 * delta, (
        f"gradient drifted by {drift:.3e} for a mass step of {delta:.3e} that "
        "crosses no MAC boundary"
    )


def test_force_scale_nodes_receives_no_cotangent():
    """``force_scale_nodes`` is a pytree leaf but must never carry a gradient.

    It is the p=1 prepass output that parameterises the frozen MAC decision. It
    is a leaf of ``FMMPreparedState``, so a future refactor could plausibly wire
    it into the differentiated seam -- where a cotangent would be meaningless,
    because the decision it drives is piecewise constant.
    """

    positions, masses, probe = _system(N_PARTICLES, seed=1)
    fmm = _mass_mac_solver()
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
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
    # the force at all, because the seam never reads them.
    baseline = fmm.differentiable_accelerations(state, positions, masses)
    perturbed = fmm.differentiable_accelerations(
        replace(state, force_scale_nodes=state.force_scale_nodes * 1.5),
        positions,
        masses,
    )
    assert np.array_equal(np.asarray(baseline), np.asarray(perturbed))


# ---------------------------------------------------------------------------
# traceability of the geometry modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["com", "tree_approx", "runtime"])
def test_device_side_geometry_modes_are_differentiable(mode):
    """The device-side MAC geometry modes must survive jit and grad."""

    positions, masses, probe = _system(N_PARTICLES, seed=1)
    fmm = _mass_mac_solver(geometry=mode)
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)

    @jax.jit
    def loss(pos, m):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, pos, m))

    grad = jax.grad(loss)(positions, masses)
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("mode", ["exact", "tree"])
def test_host_loop_geometry_modes_reject_tracers_clearly(mode):
    """The numpy host-loop modes must fail with an actionable message.

    They run a Python loop over nodes, so they cannot be traced at all. Without an
    explicit guard the failure surfaces as an opaque numpy conversion error deep
    inside the policy builder.
    """

    from jaccpot.runtime._adaptive_policy import resolve_dehnen_geometry
    from jaccpot.upward.tree_expansions import prepare_upward_sweep
    from yggdrax.tree import Tree

    positions, masses, _ = _system(N_PARTICLES, seed=1)
    tree = Tree.from_particles(
        positions, masses, leaf_size=LEAF_SIZE, tree_type="radix",
        target_leaf_particles=LEAF_SIZE, refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]
    masses_sorted = masses[tree.particle_indices]
    upward = prepare_upward_sweep(
        tree, positions_sorted, masses_sorted, max_order=MAX_ORDER, center_mode="com"
    )

    def run(pos):
        return resolve_dehnen_geometry(
            geometry_mode=mode,
            tree=tree,
            positions_sorted=pos,
            upward=upward,
            dtype=jnp.float64,
        )[1]

    with pytest.raises(RuntimeError, match="cannot be traced"):
        jax.jit(run)(positions_sorted)
