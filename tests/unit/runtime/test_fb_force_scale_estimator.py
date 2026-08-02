"""The O(N) estimator for Dehnen eq (16b)'s force scale ``f_b``.

eq (16b) replaces eq (16a)'s ``min_b |a_b|`` with ``min_b f_b``, where
``f_b = sum_{a != b} G m_a / |x_a - x_b|^2`` is the cancellation-free sum of
pairwise force magnitudes. It measurably beats (16a) -- roughly 1.5x on p99 and
2x on the tail -- but that was measured by injecting an *exact* O(N^2) ``f_b``,
which is a ceiling and not something a production path can run.

:func:`~jaccpot.runtime._adaptive_policy.estimate_particle_force_scale` computes
it in O(N) from the pair partition the traversal already built: exact scalar sums
over the near pairs, monopoles over the far ones. These tests pin the three
properties the estimate has to have, each of which had a plausible way to be
silently wrong:

* it is **exact** where the split is all near field, so the near sum, the
  self-exclusion and the leaf enumeration are right;
* it is a **lower bound** on the exact ``f_b``, which is the safe direction --
  eq (16a)'s threshold is ``eps * s``, so an over-large scale loosens acceptance
  and makes the solver faster *and* wronger, a failure no cost measurement sees;
* the **far-field term is load-bearing**. ``f_b`` is not near-field dominated: at
  N=4096 the 16 largest contributors capture a median 13% of it. A near-only
  estimator would be wrong by nearly an order of magnitude while still looking
  like a plausible force scale, so there is a test that fails if the far term is
  dropped.
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
from jaccpot.runtime._adaptive_policy import (
    estimate_particle_force_scale,
    node_span_mass,
)

LEAF_SIZE = 8
MAX_ORDER = 4
SOFTENING = 1.0e-3
PAPER_EPS = 3.0e-3


def _sample_problem(n: int = 512, *, seed: int = 20260802):
    key_pos, key_mass = jax.random.split(jax.random.PRNGKey(seed))
    positions = jax.random.normal(key_pos, (n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=jnp.float64)) + 0.5
    return positions, masses


def _traversal_cfg() -> RuntimePolicyConfig:
    return RuntimePolicyConfig(
        retain_traversal_result=True,
        retain_interactions=True,
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=131072,
            process_block=512,
            max_interactions_per_node=65536,
            max_neighbors_per_leaf=65536,
        ),
    )


def _geometric_solver(theta: float) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=theta,
        softening=SOFTENING,
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )


def _fb_solver(**kwargs) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        # theta=1.0 mirrors how paper mode is actually driven: the criterion is
        # gated by eps, not by an opening angle. The prepass underneath still needs
        # a sane angle, which is the point of mac_force_scale_prepass_theta.
        theta=1.0,
        softening=SOFTENING,
        adaptive_eps=PAPER_EPS,
        advanced=FMMAdvancedConfig(mac_type="dehnen_error", runtime=_traversal_cfg()),
        **kwargs,
    )


def _exact_force_scale(positions, masses, *, softening=SOFTENING, G=1.0):
    """Dehnen's ``f_b``, direct O(N^2). Small-N reference only."""

    pos = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    diff = pos[None, :, :] - pos[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening * softening
    contrib = mass[None, :] / dist_sq
    np.fill_diagonal(contrib, 0.0)
    return G * np.sum(contrib, axis=1)


def _estimate_on(state, positions, masses, **kwargs):
    """Run the estimator against a prepared state's own pair partition."""

    order = np.asarray(state.tree.particle_indices)
    geometry = state.upward.geometry
    return np.asarray(
        estimate_particle_force_scale(
            tree=state.tree,
            positions_sorted=jnp.asarray(positions)[order],
            masses_sorted=jnp.asarray(masses)[order],
            node_centers=geometry.center,
            node_radii=geometry.radius,
            interaction_sources=state.interactions.sources,
            interaction_targets=state.interactions.targets,
            neighbor_offsets=state.neighbor_list.offsets,
            neighbor_counts=state.neighbor_list.counts,
            neighbor_leaf_indices=state.neighbor_list.leaf_indices,
            neighbor_indices=state.neighbor_list.neighbors,
            max_leaf_size=int(state.max_leaf_size),
            softening=SOFTENING,
            **kwargs,
        )
    )


def _far_pairs(state) -> int:
    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    return int(((sources >= 0) & (targets >= 0)).sum())


# --------------------------------------------------------------------------- #
# node masses, straight from the spans
# --------------------------------------------------------------------------- #


def test_node_span_mass_is_exact_for_every_node():
    """Every node's mass must equal the mass it spans, root included.

    Deliberately not read off ``multipole_packed[:, 0]``: two upward-M2M defects
    on this branch left nodes that span particles with a zero expansion, and an
    estimator sourced from the multipoles would have inherited both while looking
    like ordinary truncation error.
    """

    positions, masses = _sample_problem(256)
    fmm = _geometric_solver(0.5)
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    order = np.asarray(state.tree.particle_indices)
    masses_sorted = np.asarray(masses)[order]

    node_mass = np.asarray(
        node_span_mass(tree=state.tree, masses_sorted=jnp.asarray(masses_sorted))
    )
    ranges = np.asarray(state.tree.node_ranges)
    for node in range(ranges.shape[0]):
        lo, hi = int(ranges[node, 0]), int(ranges[node, 1])
        expected = float(masses_sorted[lo : hi + 1].sum()) if hi >= lo else 0.0
        assert node_mass[node] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    assert node_mass[0] == pytest.approx(float(masses_sorted.sum()), rel=1e-12)


# --------------------------------------------------------------------------- #
# the near field, exactly
# --------------------------------------------------------------------------- #


def test_estimate_is_exact_when_the_partition_is_all_near_field():
    """At an opening angle that accepts nothing, the estimate must be the exact sum.

    This is the load-bearing near-field test: with no far pairs the estimator's
    output is purely its own scalar P2P sum, so an off-by-one in a leaf span, a
    missed self block, a double-counted neighbour, or a failure to exclude
    ``a == b`` all show up here as a finite discrepancy rather than as a small
    approximation error.
    """

    positions, masses = _sample_problem(512)
    fmm = _geometric_solver(0.05)
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert _far_pairs(state) == 0, "theta=0.05 must accept no far pairs"

    estimate = _estimate_on(state, positions, masses)
    exact = _exact_force_scale(positions, masses)[
        np.asarray(state.tree.particle_indices)
    ]
    np.testing.assert_allclose(estimate, exact, rtol=1e-11, atol=0.0)


def test_the_far_field_term_is_load_bearing():
    """Dropping the far term must break the estimate, not merely blunt it.

    ``f_b`` is a global quantity: in 3D the shell population grows like ``r^2 rho``
    while each contribution falls like ``1/r^2``, so every logarithmic shell
    contributes comparably. This pins the docstring's claim with a number, so a
    later "the near field dominates anyway" simplification cannot pass review by
    inspection.

    Deliberately run with more, smaller leaves than the rest of this module. The
    far field's share of ``f_b`` grows with the number of leaves, and at the module
    default (N=512, leaf 8, 64 leaves) a near-only estimate still captures 98.9% of
    the exact sum -- so the obvious configuration is precisely the one where this
    test would pass while asserting nothing. Measured near-only medians: 0.989 at
    64 leaves, 0.807 at 256, 0.534 at 512.
    """

    positions, masses = _sample_problem(2048)
    leaf_size = 4
    fmm = _geometric_solver(0.6)
    state = fmm.prepare_state(
        positions, masses, leaf_size=leaf_size, max_order=MAX_ORDER
    )
    assert _far_pairs(state) > 0

    exact = _exact_force_scale(positions, masses)[
        np.asarray(state.tree.particle_indices)
    ]
    full = _estimate_on(state, positions, masses)
    # "Near field only" is an empty far list, not a flag: this way the comparison
    # runs the same code path production does, and there is no near-only mode to
    # drift out of sync with it.
    empty = jnp.full_like(jnp.asarray(state.interactions.sources), -1)
    order = np.asarray(state.tree.particle_indices)
    geometry = state.upward.geometry
    near_only = np.asarray(
        estimate_particle_force_scale(
            tree=state.tree,
            positions_sorted=jnp.asarray(positions)[order],
            masses_sorted=jnp.asarray(masses)[order],
            node_centers=geometry.center,
            node_radii=geometry.radius,
            interaction_sources=empty,
            interaction_targets=empty,
            neighbor_offsets=state.neighbor_list.offsets,
            neighbor_counts=state.neighbor_list.counts,
            neighbor_leaf_indices=state.neighbor_list.leaf_indices,
            neighbor_indices=state.neighbor_list.neighbors,
            max_leaf_size=int(state.max_leaf_size),
            softening=SOFTENING,
        )
    )

    full_ratio = float(np.median(full / exact))
    near_ratio = float(np.median(near_only / exact))
    assert full_ratio > 0.78, f"the full estimate should be tight, got {full_ratio}"
    assert near_ratio < 0.70, (
        "a near-field-only f_b must be visibly wrong; if this passes, the far "
        f"term stopped contributing (near-only median ratio {near_ratio})"
    )
    assert full_ratio > near_ratio * 1.3, (
        "the far term must close most of the gap to the exact sum; measured "
        f"full={full_ratio:.3f} against near-only={near_ratio:.3f}"
    )


# --------------------------------------------------------------------------- #
# the lower-bound guarantee
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("theta", [0.4, 0.6, 0.8])
def test_estimate_never_exceeds_the_exact_force_scale(theta):
    """With the default inflation the estimate must bound ``f_b`` from below.

    The direction matters and is not symmetric. eq (16a) accepts when the
    estimated error falls below ``eps * s``, so an over-large ``s`` *loosens* the
    criterion: the solver accepts more pairs, runs faster, and is less accurate.
    No cost benchmark can detect that, which is why the estimator errs low by
    construction and why this is an assertion rather than a comment.
    """

    positions, masses = _sample_problem(512)
    fmm = _geometric_solver(theta)
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    estimate = _estimate_on(state, positions, masses)
    exact = _exact_force_scale(positions, masses)[
        np.asarray(state.tree.particle_indices)
    ]
    assert np.all(estimate <= exact * (1.0 + 1e-9)), (
        "estimate exceeded the exact f_b for "
        f"{int((estimate > exact * (1 + 1e-9)).sum())} particles"
    )
    assert float(np.median(estimate / exact)) > 0.85


def test_zero_inflation_is_tighter_but_is_not_a_bound():
    """``far_center_inflation=0`` trades the guarantee for accuracy, as documented.

    Pinned so the knob's meaning cannot quietly invert: the default must be the
    safe one, and the measurement setting must be the one that overshoots.
    """

    positions, masses = _sample_problem(512)
    fmm = _geometric_solver(0.7)
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    exact = _exact_force_scale(positions, masses)[
        np.asarray(state.tree.particle_indices)
    ]
    bounded = _estimate_on(state, positions, masses, far_center_inflation=1.0)
    plain = _estimate_on(state, positions, masses, far_center_inflation=0.0)

    assert np.all(bounded <= exact * (1.0 + 1e-9))
    assert np.all(plain >= bounded - 1e-12)
    assert float(np.max(plain / exact)) > 1.0, (
        "the un-inflated variant is documented as not a bound; if it never "
        "exceeds the exact value the two knob settings have collapsed"
    )


# --------------------------------------------------------------------------- #
# the production mode
# --------------------------------------------------------------------------- #


def test_paper_fb_mode_reaches_the_criterion():
    """``mac_force_scale_mode='paper_fb'`` must change acceptance, not just run.

    ``f_b`` is strictly larger than ``|a_b|`` -- it is the same sum without the
    vector cancellation -- so at fixed ``eps`` the (16b) threshold is looser and
    accepts more far pairs. A mode that computed ``f_b`` and then failed to route
    it into the criterion would still produce correct forces, so "it ran" is not
    evidence; the accept mask has to move.
    """

    positions, masses = _sample_problem(512)
    baseline = _fb_solver()
    fb = _fb_solver(mac_force_scale_mode="paper_fb")
    state_a = baseline.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    state_b = fb.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    scale_a = np.asarray(state_a.force_scale_nodes)
    scale_b = np.asarray(state_b.force_scale_nodes)
    assert not np.allclose(scale_a, scale_b), "the two force scales are identical"
    assert np.all(
        scale_b >= scale_a - 1e-12
    ), "f_b is the cancellation-free sum, so it cannot be smaller than |a_b|"
    assert _far_pairs(state_b) >= _far_pairs(state_a)
    assert fb._impl._last_force_scale_particles is not None


def test_an_evaluation_does_not_overwrite_the_fb_cache():
    """The ``|a_b|`` recorder must leave an ``f_b`` scale alone.

    ``_record_force_scale_from_evaluation`` writes the accelerations of the step
    just evaluated into the force-scale cache, which is what makes reuse mean
    anything for eq (16a). Under (16b) that would replace ``f_b`` with ``|a_b|``
    after the first evaluation -- a silent reversion to the other criterion,
    exactly the failure that made an injected ``f_b`` survive only one
    ``prepare_state``. So under ``paper_fb_cached`` the scale must be stable
    across prepare/evaluate cycles, while ``paper_cached`` is expected to drift.
    """

    positions, masses = _sample_problem(512)

    fb = _fb_solver(mac_force_scale_mode="paper_fb_cached")
    first = fb.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    jax.block_until_ready(fb.evaluate_prepared_state(first, return_potential=False))
    second = fb.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    np.testing.assert_array_equal(
        np.asarray(first.force_scale_nodes),
        np.asarray(second.force_scale_nodes),
    )

    # Control: the (16a) cached mode does pick the evaluation up, so the test
    # above cannot pass merely because nothing ever writes the cache.
    acc_mode = _fb_solver(mac_force_scale_mode="paper_cached")
    a_first = acc_mode.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    jax.block_until_ready(
        acc_mode.evaluate_prepared_state(a_first, return_potential=False)
    )
    a_second = acc_mode.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert not np.array_equal(
        np.asarray(a_first.force_scale_nodes),
        np.asarray(a_second.force_scale_nodes),
    ), "the |a_b| recorder is inert, so the f_b stability test proves nothing"


def test_the_fb_prepass_does_not_break_the_reverse_pass():
    """FD must still agree with AD with the ``f_b`` prepass in the path.

    The MAC lives entirely on the frozen side of the differentiability seam, so
    this is not evidence that the criterion is differentiable -- it is a guard that
    a *new* prepass has not leaked a tracer into an instance attribute or broken
    the reverse pass. Both are live risks here: the estimator writes
    ``_last_force_scale_particles``, and the M2M mass-conservation and force-scale
    caches on this class are exactly the kind of state that must never capture a
    tracer.
    """

    positions, masses = _sample_problem(512)
    probe = jax.random.normal(jax.random.PRNGKey(3), (512, 3), dtype=jnp.float64)
    fmm = _fb_solver(mac_force_scale_mode="paper_fb")
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert _far_pairs(state) > 0, "config must exercise the M2L reverse pass"

    def loss(x):
        return jnp.sum(probe * fmm.differentiable_accelerations(state, x, masses))

    grad = jax.grad(loss)(positions)
    assert bool(jnp.all(jnp.isfinite(grad)))
    step = 1e-6
    ad = float(jnp.sum(grad * probe))
    fd = float(
        (loss(positions + step * probe) - loss(positions - step * probe)) / (2 * step)
    )
    assert abs(fd - ad) / (abs(fd) + 1e-300) < 1e-4


def test_the_prepass_angle_is_independent_of_the_solver_theta():
    """Changing ``theta`` must not move the ``f_b`` estimate.

    Paper mode pins ``theta`` at 1.0 on the grounds that it does not gate
    acceptance. True of the criterion, false of the prepass traversal underneath
    it, whose opening angle decides how much of ``f_b`` comes from the exact near
    field and how much from the monopole approximation -- measured at theta=1.0
    the estimate degrades to a median 0.74 of the exact value against 0.997 at
    0.5. The prepass therefore resolves its own angle, and this pins that so a
    later refactor cannot quietly reconnect it to the solver's theta.
    """

    positions, masses = _sample_problem(512)
    scales = []
    for theta in (0.3, 0.7, 1.0):
        fmm = FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="real",
            theta=theta,
            softening=SOFTENING,
            adaptive_eps=PAPER_EPS,
            mac_force_scale_mode="paper_fb",
            advanced=FMMAdvancedConfig(
                mac_type="dehnen_error", runtime=_traversal_cfg()
            ),
        )
        state = fmm.prepare_state(
            positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )
        scales.append(np.asarray(state.force_scale_nodes))

    for other in scales[1:]:
        np.testing.assert_allclose(scales[0], other, rtol=1e-12, atol=0.0)

    # And the override does move it, so the invariance above is not vacuous.
    overridden = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_eps=PAPER_EPS,
        mac_force_scale_mode="paper_fb",
        mac_force_scale_prepass_theta=0.95,
        advanced=FMMAdvancedConfig(mac_type="dehnen_error", runtime=_traversal_cfg()),
    )
    state = overridden.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert not np.allclose(
        scales[0], np.asarray(state.force_scale_nodes)
    ), "mac_force_scale_prepass_theta had no effect"


def test_a_tighter_prepass_angle_gives_a_tighter_estimate():
    """The estimate must improve monotonically as the prepass far field shrinks.

    The near field is exact and the far field is a monopole lower bound, so
    moving pairs from far to near can only raise the estimate towards the exact
    value. This is the property that justifies the default of 0.5, and it fails
    if the near and far halves are ever double-counted or gapped -- either would
    break the monotonicity even though both halves look individually plausible.
    """

    positions, masses = _sample_problem(512)
    exact = _exact_force_scale(positions, masses)
    medians = []
    for prepass_theta in (0.8, 0.6, 0.4):
        fmm = FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="real",
            theta=1.0,
            softening=SOFTENING,
            adaptive_eps=PAPER_EPS,
            mac_force_scale_mode="paper_fb",
            mac_force_scale_prepass_theta=prepass_theta,
            advanced=FMMAdvancedConfig(
                mac_type="dehnen_error", runtime=_traversal_cfg()
            ),
        )
        state = fmm.prepare_state(
            positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )
        estimate = np.asarray(fmm._impl._last_force_scale_particles)
        reference = exact[np.asarray(state.tree.particle_indices)]
        medians.append(float(np.median(estimate / reference)))

    assert (
        medians[0] <= medians[1] + 1e-9 <= medians[2] + 2e-9
    ), f"estimate quality is not monotone in the prepass angle: {medians}"
    assert medians[-1] > 0.95
