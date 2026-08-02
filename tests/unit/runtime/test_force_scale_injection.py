"""The ``force_scale_nodes`` parameter and the scalar node reduction behind it.

Dehnen (2014, arXiv:1405.2255) eq (16b) is eq (16a) with ``min_b f_b`` on the
right-hand side instead of ``min_b |a_b|``, where
``f_b = sum_{a != b} G m_a / |x_a - x_b|^2`` is the cancellation-free force scale.
The criterion, the traversal and the error estimator are all unchanged, so the
whole of (16b) is a different per-node force scale -- which makes an explicit way
to supply one the natural seam for it.

Before this parameter existed the only way in was assigning
``fmm._impl._last_force_scale_nodes``, which a live writer now overwrites after
every full-order evaluation. An injected scale survived exactly one
``prepare_state``, so a prepare/evaluate loop silently reverted to (16a) while
appearing to measure (16b). These tests pin the parameter's contract, including
that it does not touch that cache.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.tree import Tree

from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    RuntimePolicyConfig,
)
from jaccpot.runtime._adaptive_policy import (
    compute_node_force_scale_from_sorted_acc,
    compute_node_force_scale_from_sorted_magnitudes,
    resolve_dehnen_geometry,
)

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


N_PARTICLES = 512
LEAF_SIZE = 8
MAX_ORDER = 4
PAPER_EPS = 3.0e-3
SOFTENING = 1.0e-3


def _sample_problem(n: int = N_PARTICLES):
    key_pos, key_mass = jax.random.split(jax.random.PRNGKey(20260801))
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


def _paper_solver(**kwargs) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_eps=PAPER_EPS,
        advanced=FMMAdvancedConfig(mac_type="dehnen_error", runtime=_traversal_cfg()),
        **kwargs,
    )


def _prepare(fmm, positions, masses, **kwargs):
    return fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER, **kwargs
    )


def _far_pairs(state) -> int:
    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    return int(((sources >= 0) & (targets >= 0)).sum())


def _exact_force_scale(positions, masses, *, softening=SOFTENING, G=1.0):
    """Dehnen's ``f_b``, direct O(N^2). Small-N reference only."""

    pos = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    diff = pos[None, :, :] - pos[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening * softening
    contrib = mass[None, :] / dist_sq
    np.fill_diagonal(contrib, 0.0)
    return G * np.sum(contrib, axis=1)


# --------------------------------------------------------------------------- #
# the scalar node reduction
# --------------------------------------------------------------------------- #


def test_scalar_reduction_matches_the_acceleration_reduction_on_magnitudes():
    """The vector entry point must be exactly the scalar one after a norm."""

    positions, masses = _sample_problem(256)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=LEAF_SIZE,
        tree_type="radix",
        target_leaf_particles=LEAF_SIZE,
        refine_local=False,
    )
    accelerations = jax.random.normal(
        jax.random.PRNGKey(5), (256, 3), dtype=jnp.float64
    )

    for reduction in ("min", "max"):
        from_acc = compute_node_force_scale_from_sorted_acc(
            tree=tree, accelerations_sorted=accelerations, reduction=reduction
        )
        from_mag = compute_node_force_scale_from_sorted_magnitudes(
            tree=tree,
            magnitudes_sorted=jnp.linalg.norm(accelerations, axis=1),
            reduction=reduction,
        )
        np.testing.assert_array_equal(np.asarray(from_acc), np.asarray(from_mag))


def test_scalar_reduction_takes_the_min_over_each_node_span():
    """Spot-check the reduction against a plain numpy min over node ranges."""

    positions, masses = _sample_problem(256)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=LEAF_SIZE,
        tree_type="radix",
        target_leaf_particles=LEAF_SIZE,
        refine_local=False,
    )
    values = (
        jnp.abs(jax.random.normal(jax.random.PRNGKey(6), (256,), dtype=jnp.float64))
        + 0.1
    )

    scales = np.asarray(
        compute_node_force_scale_from_sorted_magnitudes(
            tree=tree, magnitudes_sorted=values, reduction="min"
        )
    )

    values_np = np.asarray(values)
    node_ranges = np.asarray(tree.node_ranges)
    checked = 0
    for node in range(node_ranges.shape[0]):
        lo, hi = int(node_ranges[node, 0]), int(node_ranges[node, 1])
        if hi < lo:
            continue
        np.testing.assert_allclose(scales[node], values_np[lo : hi + 1].min())
        checked += 1
    assert checked > 0


def test_scalar_reduction_rejects_vector_input():
    """A (N, 3) array here would silently reduce over the wrong axis."""

    positions, masses = _sample_problem(64)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=LEAF_SIZE,
        tree_type="radix",
        target_leaf_particles=LEAF_SIZE,
        refine_local=False,
    )
    with pytest.raises(ValueError, match="must be 1-D"):
        compute_node_force_scale_from_sorted_magnitudes(
            tree=tree, magnitudes_sorted=jnp.zeros((64, 3)), reduction="min"
        )


# --------------------------------------------------------------------------- #
# the prepare_state parameter
# --------------------------------------------------------------------------- #


def test_supplied_force_scale_is_used_verbatim():
    positions, masses = _sample_problem()
    fmm = _paper_solver()
    node_count = int(_prepare(fmm, positions, masses).tree.parent.shape[0])
    supplied = jnp.linspace(1.0, 2.0, node_count, dtype=jnp.float64)

    state = _prepare(fmm, positions, masses, force_scale_nodes=supplied)

    np.testing.assert_array_equal(
        np.asarray(state.force_scale_nodes), np.asarray(supplied)
    )


def test_supplied_force_scale_does_not_touch_the_reuse_cache():
    """The whole point: injection must not leak into the next unqualified call.

    Seeding the cache here would make a later ``prepare_state`` that supplies
    nothing silently inherit an externally injected scale.
    """

    positions, masses = _sample_problem()
    fmm = _paper_solver()
    baseline = _prepare(fmm, positions, masses)
    cached = np.asarray(fmm._impl._last_force_scale_nodes).copy()
    node_count = int(baseline.tree.parent.shape[0])

    _prepare(
        fmm,
        positions,
        masses,
        force_scale_nodes=jnp.full((node_count,), 7.0, dtype=jnp.float64),
    )

    np.testing.assert_array_equal(np.asarray(fmm._impl._last_force_scale_nodes), cached)
    np.testing.assert_array_equal(
        np.asarray(_prepare(fmm, positions, masses).force_scale_nodes), cached
    )


def test_supplied_force_scale_skips_the_prepass(monkeypatch):
    positions, masses = _sample_problem()
    fmm = _paper_solver()
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
    node_count = int(_prepare(fmm, positions, masses).tree.parent.shape[0])
    calls[0] = 0
    fmm._impl._last_force_scale_nodes = None

    _prepare(
        fmm,
        positions,
        masses,
        force_scale_nodes=jnp.ones((node_count,), dtype=jnp.float64),
    )

    assert calls[0] == 0


@pytest.mark.parametrize("bad", ["short", "two_dim"])
def test_supplied_force_scale_rejects_a_mismatched_shape(bad):
    """A wrong length would otherwise fail deep inside the policy build."""

    positions, masses = _sample_problem()
    fmm = _paper_solver()
    node_count = int(_prepare(fmm, positions, masses).tree.parent.shape[0])
    supplied = (
        jnp.ones((node_count - 1,), dtype=jnp.float64)
        if bad == "short"
        else jnp.ones((node_count, 1), dtype=jnp.float64)
    )

    with pytest.raises(ValueError, match="force_scale_nodes must be a 1-D array"):
        _prepare(fmm, positions, masses, force_scale_nodes=supplied)


def test_supplied_force_scale_rejected_when_no_adaptive_path_would_use_it():
    """Silently ignoring it is exactly the failure this parameter removes."""

    positions, masses = _sample_problem()
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        advanced=FMMAdvancedConfig(mac_type="dehnen", runtime=_traversal_cfg()),
    )

    with pytest.raises(ValueError, match="no adaptive .*force-scale path"):
        _prepare(fmm, positions, masses, force_scale_nodes=jnp.ones((4,)))


# --------------------------------------------------------------------------- #
# eq (16b): the injected f_b actually reaches the criterion
# --------------------------------------------------------------------------- #


def test_eq_16b_force_scale_is_looser_than_eq_16a_at_equal_eps():
    """`f_b >= |a_b|` pointwise, so (16b)'s threshold is looser and accepts more.

    This is the whole mechanism of eq (16b) reaching the traversal, and it is
    also why Dehnen quotes a smaller eps for (16b) than for (16a). If the injected
    scale were being dropped, the two arms would accept identically.
    """

    positions, masses = _sample_problem()
    fmm = _paper_solver()

    state_16a = _prepare(fmm, positions, masses)
    f_b = _exact_force_scale(positions, masses)
    f_b_nodes = compute_node_force_scale_from_sorted_magnitudes(
        tree=state_16a.tree,
        magnitudes_sorted=jnp.asarray(f_b)[state_16a.tree.particle_indices],
        reduction="min",
    )
    state_16b = _prepare(fmm, positions, masses, force_scale_nodes=f_b_nodes)

    assert np.all(np.asarray(f_b_nodes) > 0.0)
    assert _far_pairs(state_16b) > _far_pairs(state_16a)


def _accepted_openings(positions, masses, *, eps: float, theta_max: float):
    """Opening angles ``(r_s + r_t) / d`` of every accepted far pair."""

    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=1.0,
        softening=SOFTENING,
        adaptive_eps=eps,
        mac_theta_max=theta_max,
        dehnen_geometry_mode="com",
        advanced=FMMAdvancedConfig(mac_type="dehnen_error", runtime=_traversal_cfg()),
    )
    state = _prepare(fmm, positions, masses)
    centers, radii = resolve_dehnen_geometry(
        geometry_mode="com",
        tree=state.tree,
        positions_sorted=state.positions_sorted,
        upward=state.upward,
        dtype=jnp.float64,
    )
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    keep = (sources >= 0) & (targets >= 0)
    sources, targets = sources[keep], targets[keep]
    if sources.size == 0:
        return np.zeros((0,), dtype=np.float64)
    distance = np.linalg.norm(centers[sources] - centers[targets], axis=1)
    return (radii[sources] + radii[targets]) / np.maximum(distance, 1e-300)


def test_eq_16a_admits_pairs_at_the_convergence_boundary():
    """eq (16a)'s only geometric guard is ``theta < 1``, the divergence boundary.

    ``mac_theta_max`` exists because acceptance really does reach openings where a
    truncated expansion has O(1) error; measured up to 0.999 here and on Plummer
    at N=4096. The cap is a disclosed deviation from the paper, so this pins both
    that it is needed and that it binds exactly.

    Note this is a statement about the *admitted set*, not about the eq (15) bound
    misbehaving: the bound is monotonically increasing in opening (1.7e-8 at
    opening 0.2 rising to 1.6e-1 at 0.995 for a fixed source), so what reaches the
    boundary are pairs whose own source mass and multipole power keep the estimate
    small there.
    """

    positions, masses = _sample_problem()

    uncapped = _accepted_openings(positions, masses, eps=1.0e-2, theta_max=1.0)
    assert uncapped.size > 0
    assert uncapped.max() > 0.9

    capped = _accepted_openings(positions, masses, eps=1.0e-2, theta_max=0.7)
    assert capped.size > 0
    assert capped.max() <= 0.7 + 1e-9


def test_exact_f_b_dominates_the_acceleration_magnitude_per_particle():
    """Sanity on the reference itself: no cancellation means f_b >= |a_b|.

    Guards the direction of the whole eq (16b) argument -- if this inverted, `f_b`
    would not be the cancellation-free scale the criterion assumes.
    """

    positions, masses = _sample_problem(256)
    pos = np.asarray(positions)
    mass = np.asarray(masses)
    diff = pos[None, :, :] - pos[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + SOFTENING * SOFTENING
    inv_cube = dist_sq ** (-1.5)
    np.fill_diagonal(inv_cube, 0.0)
    acc = np.einsum("ij,ij,ijk->ik", mass[None, :], inv_cube, diff)

    f_b = _exact_force_scale(positions, masses, softening=SOFTENING)

    assert np.all(f_b >= np.linalg.norm(acc, axis=1) - 1e-12)
